from __future__ import annotations
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import pathlib
from pathlib import Path
import requests
import time
from typing import Optional

import imageio
import numpy as np
import tqdm
import tyro
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from model2libero_interface import ModelClient


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def _safe_close_env(env, task_label: str) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception as exc:
        logging.warning("Failed to close LIBERO env for task `%s`: %s", task_label, exc)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224, 224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Options: libero_spatial, libero_object, libero_goal, libero_10
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1  # LIBERO-plus: each task has only 1 initial state

    #################################################################################################################
    # Utils
    #################################################################################################################
    output_dir: str = "experiments/libero_plus"  # Base output directory
    video_out_path: str = ""  # Path to save videos; auto-generated from output_dir if empty
    log_path: str = ""  # Path to save logs; auto-generated from output_dir if empty

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"

    unnorm_key: Optional[str] = None

    start_task_id: int = 0   # resume support: start from the specified task_id; 0 means from the beginning
    success_count: int = 0   # resume support: absolute success count from prior tasks; only used when start_task_id > 0


def eval_libero(args: Args) -> None:
    # Derive folder_name: "run_root/run_id/ckpt_stem"
    # e.g. .../0401_all/ABot_M0_cosmos/checkpoints/steps_40000_pytorch_model.pt
    #   -> "0401_all_ABot_M0_cosmos_steps_40000_pytorch_model"
    if args.pretrained_path:
        parts = Path(args.pretrained_path).parts
        ckpt_stem = Path(parts[-1]).stem          # steps_40000_pytorch_model
        run_id = parts[-3] if len(parts) >= 3 else ""          # ABot_M0_cosmos
        run_root = parts[-4] if len(parts) >= 4 else ""        # 0401_all
        folder_name = "_".join(filter(None, [run_root, run_id, ckpt_stem]))
    else:
        folder_name = "unknown"

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_path if args.log_path else f"{args.output_dir}/logs/{timestamp}"
    video_out_path = args.video_out_path if args.video_out_path else f"{args.output_dir}/{args.task_suite_name}/{folder_name}_{timestamp}"

    Path(log_path).mkdir(parents=True, exist_ok=True)
    Path(video_out_path).mkdir(parents=True, exist_ok=True)

    # Setup logging to both file and console
    log_file = os.path.join(log_path, f"{args.task_suite_name}.log")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for h in [logging.FileHandler(log_file), logging.StreamHandler()]:
        h.setFormatter(fmt)
        root_logger.addHandler(h)

    logging.info(f"Log dir: {log_path}")
    logging.info(f"Video dir: {video_out_path}")
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path,  # to get unnormalization stats
        unnorm_key=args.unnorm_key,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    # Load LIBERO-plus task classification for per-category disturb stats.
    # benchmark_root points to LIBERO-plus/libero/libero, so two levels up is the repo root.
    benchmark_root = get_libero_path("benchmark_root")
    task_classification_file = os.path.join(benchmark_root, "benchmark", "task_classification.json")
    disturb_res = {}
    ID2CATEGORY = {}
    with open(task_classification_file) as f:
        TASK_MAPPING = json.load(f)[args.task_suite_name]
    for item in TASK_MAPPING:
        category = item["category"]
        item_name = item["name"]
        ID2CATEGORY[item["id"]] = (category, item_name)
        if category not in disturb_res:
            disturb_res[category] = {"total_count": 0, "success_count": 0}
        disturb_res[category]["total_count"] += 1

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(args.start_task_id, num_tasks_in_suite)):  # Supports resume
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        env = None
        task_description = task.language
        try:
            # Initialize LIBERO environment and task description
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info(f"\nTask: {task_description}")

                # Reset environment
                client_model.reset(task_description=task_description)
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []
                full_actions = []

                logging.info(f"Starting episode {task_episodes + 1}...")
                step = 0

                while t < max_steps + args.num_steps_wait:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # IMPORTANT: rotate 180 degrees to match ABot train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    # Cosmos expects normal orientation (flipud of OpenGL output)
                    cosmos_img = np.ascontiguousarray(
                        np.asarray(
                            Image.fromarray(np.flipud(obs["agentview_image"])).resize((224, 224), Image.BILINEAR)
                        )
                    )
                    cosmos_wrist_img = np.ascontiguousarray(
                        np.asarray(
                            Image.fromarray(np.flipud(obs["robot0_eye_in_hand_image"])).resize(
                                (224, 224), Image.BILINEAR
                            )
                        )
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    example_dict = {
                        "image": [img, wrist_img],
                        "cosmos_image": [cosmos_img, cosmos_wrist_img],
                        "lang": str(task_description),
                        # proprio=None: matches precompute convention, avoids distribution gap
                        "proprio": None,
                    }

                    response = client_model.step(example=example_dict, step=step)

                    raw_action = response["raw_action"]

                    world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                    rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                    open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                    gripper = _binarize_gripper_open(open_gripper)

                    if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                        logging.warning(
                            f"Unexpected action sizes: "
                            f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                            f"Falling back to LIBERO_DUMMY_ACTION."
                        )
                        raise ValueError(
                            f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                            f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                        )
                    else:
                        delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                    full_actions.append(delta_action)

                    obs, reward, done, info = env.step(delta_action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        disturb_res[ID2CATEGORY[task_id + 1][0]]["success_count"] += 1
                        break
                    t += 1
                    step += 1

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                suffix = "success" if done else "failure"
                imageio.mimwrite(
                    pathlib.Path(video_out_path)
                    / f"rollout_{ID2CATEGORY[task_id + 1][1]}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=25,
                )

                full_actions = np.stack(full_actions)

                # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(
                    f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
                )

            # Log final results
            logging.info(
                f"Current task success rate: {float(task_successes) / float(task_episodes)}"
            )
            logging.info(
                f"Current total success rate: {float(total_successes) / float(total_episodes)}"
            )
        finally:
            _safe_close_env(env, task.name)

    with open(os.path.join(log_path, f"{args.task_suite_name}.json"), "w", encoding="utf-8") as f:
        json.dump(disturb_res, f)
    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")

    # Resume support: if start_task_id is provided, merge with previous results to report the overall success rate
    if args.start_task_id > 0:
        _PREV_TASKS = args.start_task_id
        _PREV_SUCCESSES = args.success_count
        _TOTAL_TASKS = num_tasks_in_suite
        _COMBINED_SUCCESSES = _PREV_SUCCESSES + total_successes
        _COMBINED_RATE = _COMBINED_SUCCESSES / _TOTAL_TASKS
        logging.info("=" * 60)
        logging.info(f"[Resume] Previous segment (task 0~{_PREV_TASKS-1}): {_PREV_TASKS} tasks, {_PREV_SUCCESSES} successes (rate={_PREV_SUCCESSES/_PREV_TASKS:.4f})")
        logging.info(f"[Resume] Current segment (task {args.start_task_id}~{_TOTAL_TASKS-1}): {total_episodes} tasks, {total_successes} successes (rate={total_successes/total_episodes:.4f})")
        logging.info(f"[Resume] Combined (all {_TOTAL_TASKS} tasks): {_COMBINED_SUCCESSES} successes (rate={_COMBINED_RATE:.4f})")
        logging.info("=" * 60)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    eval_libero(tyro.cli(Args))
