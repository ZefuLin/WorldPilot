"""
Verify precomputed Cosmos cache files for completeness and correctness.

Usage:
    python -m cosmos_bridge.verify_cache \
        --data_root /path/to/libero_lerobot \
        --cache_dir /path/to/cosmos_cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def verify_dataset(data_root: Path, cache_dir: Path, dataset_name: str) -> dict:
    """Verify one dataset's cosmos cache. Returns stats dict."""
    ds_dir = data_root / dataset_name
    cache_ds_dir = cache_dir / dataset_name

    # Load episode metadata
    episodes = []
    with open(ds_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))

    total = len(episodes)
    missing = []
    bad_shape = []
    bad_value = []
    ok = 0

    for ep in episodes:
        ep_idx = ep["episode_index"]
        ep_len = ep["length"]
        npz_path = cache_ds_dir / f"episode_{ep_idx:06d}.npz"

        if not npz_path.exists():
            missing.append(ep_idx)
            continue

        try:
            data = np.load(npz_path)

            # Check keys exist
            for key in ["future_image_latents", "action_chunk", "value"]:
                if key not in data:
                    bad_shape.append((ep_idx, f"missing key: {key}"))
                    continue

            lat = data["future_image_latents"]
            act = data["action_chunk"]
            val = data["value"]

            # Check shapes
            errors = []
            if lat.shape != (ep_len, 2, 16, 28, 28):
                errors.append(f"latents shape {lat.shape} != expected ({ep_len}, 2, 16, 28, 28)")
            if act.shape != (ep_len, 16, 7):
                errors.append(f"action shape {act.shape} != expected ({ep_len}, 16, 7)")
            if val.shape != (ep_len,):
                errors.append(f"value shape {val.shape} != expected ({ep_len},)")

            # Check dtypes
            if lat.dtype != np.float16:
                errors.append(f"latents dtype {lat.dtype} != float16")
            if act.dtype != np.float16:
                errors.append(f"action dtype {act.dtype} != float16")
            if val.dtype != np.float32:
                errors.append(f"value dtype {val.dtype} != float32")

            # Check for NaN/Inf
            if np.any(np.isnan(lat)) or np.any(np.isinf(lat)):
                errors.append("latents contain NaN/Inf")
            if np.any(np.isnan(act)) or np.any(np.isinf(act)):
                errors.append("action contain NaN/Inf")
            if np.any(np.isnan(val)) or np.any(np.isinf(val)):
                errors.append("value contain NaN/Inf")

            # Check value range (should be [0, 1])
            if val.min() < -0.1 or val.max() > 1.1:
                bad_value.append((ep_idx, f"value range [{val.min():.4f}, {val.max():.4f}]"))

            if errors:
                bad_shape.append((ep_idx, "; ".join(errors)))
            else:
                ok += 1

        except Exception as e:
            bad_shape.append((ep_idx, f"load error: {e}"))

    return {
        "dataset": dataset_name,
        "total_episodes": total,
        "ok": ok,
        "missing": missing,
        "bad_shape": bad_shape,
        "bad_value": bad_value,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify Cosmos precomputed cache")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="*", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)

    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = sorted(
            d.name for d in data_root.iterdir()
            if d.is_dir() and d.name.startswith("libero_") and d.name.endswith("_lerobot")
        )

    all_ok = True
    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Verifying: {ds_name}")
        print(f"{'='*60}")

        result = verify_dataset(data_root, cache_dir, ds_name)

        print(f"  Episodes: {result['ok']}/{result['total_episodes']} OK")

        if result["missing"]:
            all_ok = False
            print(f"  ❌ Missing ({len(result['missing'])}): {result['missing'][:10]}"
                  + ("..." if len(result["missing"]) > 10 else ""))

        if result["bad_shape"]:
            all_ok = False
            for ep_idx, err in result["bad_shape"][:5]:
                print(f"  ❌ ep{ep_idx:06d}: {err}")
            if len(result["bad_shape"]) > 5:
                print(f"  ... and {len(result['bad_shape']) - 5} more")

        if result["bad_value"]:
            for ep_idx, err in result["bad_value"][:5]:
                print(f"  ⚠️  ep{ep_idx:06d}: {err}")

        if not result["missing"] and not result["bad_shape"]:
            print(f"  ✅ All {result['total_episodes']} episodes verified")

    print(f"\n{'='*60}")
    if all_ok:
        print("✅ All datasets verified successfully!")
    else:
        print("❌ Some datasets have issues. Fix missing/broken files and re-run precompute.")
        sys.exit(1)


if __name__ == "__main__":
    main()
