"""
Precompute Cosmos predictions for all LIBERO frames.

Iterates over every episode/frame in the LIBERO lerobot datasets,
queries the running Cosmos Server via WebSocket, and saves the results
as per-episode .npz files.

Usage:
    # 1. Start Cosmos Server(s) on GPU(s)
    CUDA_VISIBLE_DEVICES=6 python -m cosmos_bridge.cosmos_server \
        --cosmos_ckpt ... --cosmos_dataset_stats ... --cosmos_t5_embeddings ... --port 8002

    # 2. Run precompute
    python -m cosmos_bridge.precompute \
        --data_root /path/to/libero_lerobot \
        --output_dir /path/to/cosmos_cache \
        --cosmos_host 127.0.0.1 --cosmos_port 8002 \
        --num_workers 4

Supports resume: existing .npz files are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers (lightweight, no torch dependency)
# ---------------------------------------------------------------------------

def _load_episodes_meta(dataset_dir: Path) -> List[Dict]:
    """Load episodes.jsonl → list of {episode_index, tasks, length}."""
    episodes = []
    with open(dataset_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def _read_video_frames(dataset_dir: Path, episode_idx: int, video_key: str, num_frames: int):
    """Read all frames from an episode video file. Returns list of (H,W,3) uint8 arrays."""
    import av

    chunk_idx = episode_idx // 1000
    video_path = (
        dataset_dir / "videos" / f"chunk-{chunk_idx:03d}" / video_key
        / f"episode_{episode_idx:06d}.mp4"
    )

    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= num_frames:
                break
    return frames


# ---------------------------------------------------------------------------
# Worker: process one episode
# ---------------------------------------------------------------------------

def _process_episode(args_tuple):
    """Process a single episode: read frames, query Cosmos, save npz."""
    dataset_dir_str, dataset_name, episode_idx, episode_length, task_text, \
        output_dir_str, cosmos_host, cosmos_port = args_tuple

    dataset_dir = Path(dataset_dir_str)
    output_dir = Path(output_dir_str)
    out_path = output_dir / dataset_name / f"episode_{episode_idx:06d}.npz"

    if out_path.exists():
        return f"SKIP {dataset_name}/ep{episode_idx:06d}"

    from cosmos_bridge.cosmos_client import CosmosClient
    from PIL import Image

    try:
        num_frames = episode_length

        # Read video frames
        primary_frames = _read_video_frames(
            dataset_dir, episode_idx, "observation.images.image", num_frames
        )
        wrist_frames = _read_video_frames(
            dataset_dir, episode_idx, "observation.images.wrist_image", num_frames
        )

        # Ensure we have the right number of frames
        actual_frames = min(len(primary_frames), len(wrist_frames), num_frames)

        # Connect to Cosmos Server
        client = CosmosClient(host=cosmos_host, port=cosmos_port)

        all_latents = []
        all_actions = []
        all_values = []

        t0 = time.time()
        for fi in range(actual_frames):
            primary = np.asarray(
                Image.fromarray(primary_frames[fi]).resize((224, 224))
            )
            wrist = np.asarray(
                Image.fromarray(wrist_frames[fi]).resize((224, 224))
            )

            result = client.query_single(
                primary, wrist, task_text, proprio=None, flip_images=False
            )

            all_latents.append(result["future_image_latents"].astype(np.float16))
            all_actions.append(result["action_chunk"].astype(np.float16))
            all_values.append(np.float32(result["value"]))

        client.close()
        elapsed = time.time() - t0

        # Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            future_image_latents=np.stack(all_latents),   # (T, 2, 16, 28, 28) f16
            action_chunk=np.stack(all_actions),             # (T, 16, 7) f16
            value=np.array(all_values, dtype=np.float32),   # (T,) f32
        )

        fps = actual_frames / elapsed if elapsed > 0 else 0
        return f"OK   {dataset_name}/ep{episode_idx:06d} ({actual_frames} frames, {elapsed:.1f}s, {fps:.1f} fps)"

    except Exception as e:
        log.exception(f"FAIL {dataset_name}/ep{episode_idx:06d}")
        return f"FAIL {dataset_name}/ep{episode_idx:06d}: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Precompute Cosmos predictions for LIBERO")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root dir containing LIBERO lerobot datasets")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output dir for cosmos cache npz files")
    parser.add_argument("--cosmos_host", type=str, default="127.0.0.1")
    parser.add_argument("--cosmos_port", type=int, default=8002)
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel worker processes")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Specific dataset names (default: all libero_*_lerobot)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover datasets
    if args.datasets:
        dataset_dirs = [data_root / d for d in args.datasets]
    else:
        dataset_dirs = sorted(data_root.glob("libero_*_lerobot"))

    # Build task list
    tasks: List[Tuple] = []
    total_frames = 0
    for ds_dir in dataset_dirs:
        if not ds_dir.is_dir():
            log.warning(f"Skipping: {ds_dir}")
            continue
        ds_name = ds_dir.name
        episodes = _load_episodes_meta(ds_dir)
        for ep in episodes:
            ep_idx = ep["episode_index"]
            ep_len = ep["length"]
            task_text = ep["tasks"][0] if ep.get("tasks") else "manipulate the object"
            tasks.append((
                str(ds_dir), ds_name, ep_idx, ep_len, task_text,
                str(output_dir), args.cosmos_host, args.cosmos_port,
            ))
            total_frames += ep_len

    log.info(f"Datasets: {len(dataset_dirs)}, Episodes: {len(tasks)}, Frames: {total_frames}")

    # Count already done
    done = sum(
        1 for t in tasks
        if (Path(t[5]) / t[1] / f"episode_{t[2]:06d}.npz").exists()
    )
    log.info(f"Completed: {done}/{len(tasks)}, Remaining: {len(tasks) - done}")

    if args.num_workers <= 1:
        for i, task in enumerate(tasks):
            result = _process_episode(task)
            log.info(f"[{i+1}/{len(tasks)}] {result}")
    else:
        with mp.Pool(processes=args.num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_process_episode, tasks)):
                log.info(f"[{i+1}/{len(tasks)}] {result}")

    log.info("Precompute complete!")


if __name__ == "__main__":
    main()
