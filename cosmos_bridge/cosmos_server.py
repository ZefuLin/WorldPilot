"""
Cosmos Policy WebSocket server for ABot integration.

Runs in the `cosmos-policy` conda environment. Receives raw env images (+optional
proprio), runs Cosmos inference via cosmos_utils.get_action(), and returns:
  - action_chunk:          (16, 7) float32, normalized [-1,1]
  - future_image_latents:  (num_future_views, 16, 28, 28) float32
  - future_primary:        (H, W, 3) uint8  decoded future primary image
  - future_wrist:          (H, W, 3) uint8  decoded future wrist image
  - value:                 float in [0, 1]

Usage:
    cd /path/to/WorldPilot
    CUDA_VISIBLE_DEVICES=0 mamba run -n cosmos-policy python -m cosmos_bridge.cosmos_server \\
        --cosmos_config cosmos_predict2_2b_480p_libero__inference_only \\
        --cosmos_ckpt /path/to/Cosmos-Policy-LIBERO-Predict2-2B.pt \\
        --cosmos_dataset_stats /path/to/libero_dataset_statistics.json \\
        --cosmos_t5_embeddings /path/to/libero_t5_embeddings.pkl \\
        --port 8002
"""

from __future__ import annotations

import argparse
import asyncio
import os
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

FUTURE_WRIST_LATENT_IDX = 6
FUTURE_PRIMARY_LATENT_IDX = 7


@dataclass
class CosmosCfg:
    """Minimal config that mimics fields expected by cosmos_utils.get_action."""
    suite: str = "libero"
    config: str = ""
    ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"
    use_wrist_image: bool = True
    use_third_person_image: bool = True
    num_wrist_images: int = 1
    num_third_person_images: int = 1
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = False
    trained_with_image_aug: bool = True
    use_jpeg_compression: bool = True
    chunk_size: int = 16
    action_dim: int = 7
    proprio_dim: int = 9
    use_variance_scale: bool = False
    num_denoising_steps_action: int = 5
    flip_images: bool = True


class CosmosAdvisorPolicy:
    """Wraps Cosmos inference for the WebSocket server route."""

    def __init__(self, cfg: CosmosCfg, model, dataset_stats):
        self.cfg = cfg
        self.model = model
        self.dataset_stats = dataset_stats

    def predict_action(self, **msg) -> Dict[str, Any]:
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_action,
            extract_action_chunk_from_latent_sequence,
            extract_value_from_latent_sequence,
        )

        primary = np.asarray(msg["primary"], dtype=np.uint8)
        lang = str(msg["lang"])

        # flip_images: only needed when images come from LIBERO env (OpenGL, y-up).
        # ABot DataLoader outputs normal-orientation images → caller sends flip_images=False (default).
        # LIBERO eval sends flip_images=True.
        flip = msg.get("flip_images", False)
        if flip:
            primary = np.ascontiguousarray(np.flipud(primary))

        wrist = np.asarray(msg["wrist"], dtype=np.uint8)
        if flip:
            wrist = np.ascontiguousarray(np.flipud(wrist))
        obs = {
            "primary_image": primary,
            "wrist_image": wrist,
        }
        if msg.get("proprio") is not None:
            obs["proprio"] = np.asarray(msg["proprio"], dtype=np.float32)
        else:
            # Cosmos requires proprio; use zeros when not provided.
            obs["proprio"] = np.zeros(self.cfg.proprio_dim, dtype=np.float32)

        result = get_action(
            self.cfg,
            self.model,
            self.dataset_stats,
            obs,
            task_label_or_embedding=lang,
            num_denoising_steps_action=self.cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=True,
        )

        gen_latent = result["generated_latent"]
        latent_indices = result.get("latent_indices", {})

        future_latents = []
        for key in (
            "future_wrist_image_latent_idx",
            "future_wrist_image2_latent_idx",
            "future_image_latent_idx",
        ):
            idx = int(latent_indices.get(key, -1))
            if idx >= 0:
                future_latents.append(gen_latent[0, :, idx].cpu().numpy())
        if not future_latents:
            future_latents = [
                gen_latent[0, :, FUTURE_WRIST_LATENT_IDX].cpu().numpy(),
                gen_latent[0, :, FUTURE_PRIMARY_LATENT_IDX].cpu().numpy(),
            ]

        if "actions" in result:
            raw_actions = np.asarray(result["actions"], dtype=np.float32)
            if raw_actions.ndim == 3:
                raw_actions = raw_actions[0]
        else:
            action_indices = torch.full(
                (gen_latent.shape[0],),
                int(latent_indices.get("action_latent_idx", 4)),
                dtype=torch.int64,
                device=gen_latent.device,
            )
            raw_actions = (
                extract_action_chunk_from_latent_sequence(
                    gen_latent,
                    action_shape=(self.cfg.chunk_size, self.cfg.action_dim),
                    action_indices=action_indices,
                )
                .to(torch.float32)
                .cpu()
                .numpy()
            )[0]

        value_indices = torch.full(
            (gen_latent.shape[0],), -1, dtype=torch.int64, device=gen_latent.device
        )
        value_pred = extract_value_from_latent_sequence(gen_latent, value_indices)
        value_pred = ((value_pred + 1) / 2).clamp(0, 1).cpu().item()

        future_preds = result.get("future_image_predictions", {}) if isinstance(result, dict) else {}
        future_primary_img = future_preds.get("future_image", np.zeros((1,), dtype=np.uint8))
        future_wrist_img = future_preds.get("future_wrist_image", np.zeros((1,), dtype=np.uint8))
        future_wrist2_img = future_preds.get("future_wrist_image2", np.zeros((1,), dtype=np.uint8))

        return {
            "action_chunk": raw_actions.astype(np.float32),
            "future_image_latents": np.stack(future_latents).astype(np.float32),
            "future_primary": np.asarray(future_primary_img, dtype=np.uint8),
            "future_wrist": np.asarray(future_wrist_img, dtype=np.uint8),
            "future_wrist2": np.asarray(future_wrist2_img, dtype=np.uint8),
            "value": float(value_pred),
        }


def _get_args():
    p = argparse.ArgumentParser(description="Cosmos Advisor WebSocket Server")
    p.add_argument("--cosmos_root", type=str, default=os.environ.get("COSMOS_ROOT", ""))
    p.add_argument("--cosmos_config", type=str,
                    default="cosmos_predict2_2b_480p_libero__inference_only")
    p.add_argument("--cosmos_ckpt", type=str, required=True)
    p.add_argument("--cosmos_config_file", type=str,
                    default="cosmos_policy/config/config.py")
    p.add_argument("--cosmos_dataset_stats", type=str, required=True)
    p.add_argument("--cosmos_t5_embeddings", type=str, required=True)
    p.add_argument("--suite", type=str, default="auto", choices=["auto", "libero"])
    p.add_argument("--chunk_size", type=int, default=None)
    p.add_argument("--action_dim", type=int, default=None)
    p.add_argument("--proprio_dim", type=int, default=None)
    p.add_argument("--num_denoising_steps", type=int, default=5)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8002)
    return p.parse_args()


def _resolve_runtime_cfg(args) -> tuple[str, int, int, int, int, int]:
    suite = args.suite
    config_name = args.cosmos_config.lower()
    if suite == "auto":
        if "libero" not in config_name:
            raise ValueError(
                "This public evaluation release only supports LIBERO/LIBERO-plus Cosmos configs."
            )
        suite = "libero"

    default_chunk, default_action_dim, default_proprio_dim = 16, 7, 9

    chunk_size = args.chunk_size if args.chunk_size is not None else default_chunk
    action_dim = args.action_dim if args.action_dim is not None else default_action_dim
    proprio_dim = args.proprio_dim if args.proprio_dim is not None else default_proprio_dim
    num_wrist_images = 1
    num_third_person_images = 1
    return suite, chunk_size, action_dim, proprio_dim, num_wrist_images, num_third_person_images


def _fix_degenerate_stats(dataset_stats: dict) -> dict:
    """Avoid division by zero for padded dimensions whose min equals max."""
    for prefix in ("actions", "proprio"):
        min_key = f"{prefix}_min"
        max_key = f"{prefix}_max"
        if min_key not in dataset_stats or max_key not in dataset_stats:
            continue
        mn = np.asarray(dataset_stats[min_key])
        mx = np.asarray(dataset_stats[max_key])
        mask = mn == mx
        if np.any(mask):
            fixed = mx.copy()
            fixed[mask] = mn[mask] + 1.0
            dataset_stats[max_key] = fixed
            log.info("Adjusted %d degenerate %s stats dimensions", int(mask.sum()), prefix)
    return dataset_stats


def main():
    args = _get_args()
    suite, chunk_size, action_dim, proprio_dim, num_wrist_images, num_third_person_images = _resolve_runtime_cfg(args)

    if args.cosmos_root and args.cosmos_root not in sys.path:
        sys.path.insert(0, args.cosmos_root)

    abot_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    if abot_root not in sys.path:
        sys.path.insert(0, abot_root)

    from cosmos_policy.experiments.robot import cosmos_utils
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    log.info("Loading Cosmos model: config=%s", args.cosmos_config)
    model, _ = load_model_from_checkpoint(
        experiment_name=args.cosmos_config,
        s3_checkpoint_dir=args.cosmos_ckpt,
        config_file=args.cosmos_config_file,
        load_ema_to_reg=False,
    )
    model.eval()
    model = model.to("cuda")
    log.info("Cosmos model loaded.")

    cosmos_utils.ACTION_DIM = action_dim
    dataset_stats = _fix_degenerate_stats(cosmos_utils.load_dataset_stats(args.cosmos_dataset_stats))
    cosmos_utils.init_t5_text_embeddings_cache(args.cosmos_t5_embeddings)

    cfg = CosmosCfg(
        suite=suite,
        config=args.cosmos_config,
        ckpt_path=args.cosmos_ckpt,
        config_file=args.cosmos_config_file,
        num_wrist_images=num_wrist_images,
        num_third_person_images=num_third_person_images,
        chunk_size=chunk_size,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        num_denoising_steps_action=args.num_denoising_steps,
    )
    log.info(
        "Runtime cfg: suite=%s chunk_size=%d action_dim=%d proprio_dim=%d num_wrist=%d",
        cfg.suite,
        cfg.chunk_size,
        cfg.action_dim,
        cfg.proprio_dim,
        cfg.num_wrist_images,
    )

    policy = CosmosAdvisorPolicy(cfg, model, dataset_stats)

    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=-1,
        metadata={"env": "cosmos_advisor", "chunk_size": chunk_size, "action_dim": action_dim},
    )
    log.info("Cosmos Advisor Server listening on ws://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
