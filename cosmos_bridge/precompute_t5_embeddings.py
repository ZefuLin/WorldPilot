"""
Pre-compute T5 embeddings for ALL LIBERO-plus task instructions.

Enumerates every task across all LIBERO-plus benchmark suites (spatial,
object, goal, 10), including table/tb/view/add/language variants.
Loads the existing T5 pkl cache, computes any missing embeddings, and
saves the updated cache.

Must run in the cosmos-policy conda environment (for T5 encoder) with
LIBERO-plus available (for BDDL parsing of _language_ variants).

Usage (from repo root):
    LIBERO_PLUS_ROOT=/path/to/LIBERO-plus \
    LIBERO_CONFIG_PATH=~/.libero-plus \
    CUDA_VISIBLE_DEVICES=0 mamba run -n cosmos-policy \
        python -m cosmos_bridge.precompute_t5_embeddings \
        --pkl /path/to/libero_t5_embeddings.pkl
"""

from __future__ import annotations

import argparse
import ast
import os
import pickle
import shutil
import time
from pathlib import Path

import torch

_libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT", "").strip()
LIBERO_PLUS_ROOT = Path(_libero_plus_root) if _libero_plus_root else None
TASK_MAP_FILE = LIBERO_PLUS_ROOT / "libero" / "libero" / "benchmark" / "libero_suite_task_map.py" if LIBERO_PLUS_ROOT else None
BDDL_UTILS_FILE = LIBERO_PLUS_ROOT / "libero" / "libero" / "envs" / "bddl_utils.py" if LIBERO_PLUS_ROOT else None
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def _load_task_map() -> dict:
    """Parse libero_task_map from source without importing the benchmark package."""
    with open(TASK_MAP_FILE) as f:
        content = f.read()
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "libero_task_map":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"Could not find libero_task_map in {TASK_MAP_FILE}")


def _get_bddl_language(bddl_file_path: str) -> str:
    """Extract language_instruction from a BDDL file without importing libero/robosuite.

    Directly imports only bddl_utils.get_problem_info by loading the file as a module,
    avoiding the heavy transitive import chain.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_bddl_utils", BDDL_UTILS_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    info = mod.get_problem_info(bddl_file_path)
    return info["language_instruction"]


# Cache the bddl_utils module to avoid re-loading for every file
_bddl_mod = None

def _get_bddl_language_cached(bddl_file_path: str) -> str:
    global _bddl_mod
    if _bddl_mod is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_bddl_utils", BDDL_UTILS_FILE)
        _bddl_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_bddl_mod)
    info = _bddl_mod.get_problem_info(bddl_file_path)
    return info["language_instruction"]


_bddl_root_cache: str | None = None

def _get_libero_bddl_root(libero_config_dir: str) -> str:
    """Resolve bddl_files path from LIBERO config (supports .yaml and .cfg)."""
    global _bddl_root_cache
    if _bddl_root_cache is not None:
        return _bddl_root_cache

    yaml_file = os.path.join(libero_config_dir, "config.yaml")
    cfg_file = os.path.join(libero_config_dir, "libero.cfg")

    if os.path.exists(yaml_file):
        import yaml
        with open(yaml_file) as f:
            cfg = yaml.safe_load(f)
        _bddl_root_cache = cfg.get("bddl_files")
    elif os.path.exists(cfg_file):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(cfg_file)
        _bddl_root_cache = cfg.get("libero", "bddl_files", fallback=None)

    if _bddl_root_cache is None:
        raise RuntimeError(
            f"Cannot find bddl_files path in {yaml_file} or {cfg_file}. "
            f"Set LIBERO_CONFIG_PATH to the directory containing config.yaml."
        )
    return _bddl_root_cache


def _filename_to_language(suite_name: str, task_name: str) -> str:
    """Convert task filename to language instruction, matching LIBERO-plus logic."""
    bddl_name = task_name + ".bddl"

    if "_language_" not in task_name:
        # Direct filename -> space-separated string
        if bddl_name[0].isupper():
            if "SCENE10" in bddl_name:
                language = " ".join(bddl_name[bddl_name.find("SCENE") + 8:].split("_"))
            else:
                language = " ".join(bddl_name[bddl_name.find("SCENE") + 7:].split("_"))
        else:
            language = " ".join(bddl_name.split("_"))
        en = language.find(".bddl")
        return language[:en]
    else:
        # _language_ variants: parse BDDL file for paraphrased instruction
        libero_config = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero-plus"))
        bddl_root = _get_libero_bddl_root(libero_config)

        if "_view_" in task_name:
            bddl_path = os.path.join(bddl_root, suite_name, task_name.split("_view_")[0] + ".bddl")
        else:
            bddl_path = os.path.join(bddl_root, suite_name, bddl_name)

        return _get_bddl_language_cached(bddl_path)


def collect_all_instructions() -> list[str]:
    """Enumerate every unique task instruction across all LIBERO-plus benchmark suites."""
    task_map = _load_task_map()
    all_instructions: set[str] = set()
    lang_variant_count = 0

    for suite_name in SUITES:
        tasks = task_map[suite_name]
        lang_in_suite = 0
        for task_name in tasks:
            lang = _filename_to_language(suite_name, task_name)
            all_instructions.add(lang)
            if "_language_" in task_name:
                lang_in_suite += 1
        lang_variant_count += lang_in_suite
        print(f"  {suite_name}: {len(tasks)} tasks ({lang_in_suite} _language_ variants)")

    print(f"  Total _language_ variants parsed from BDDL: {lang_variant_count}")
    return sorted(all_instructions)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute T5 embeddings for LIBERO-plus")
    parser.add_argument("--pkl", type=str, required=True,
                        help="Path to T5 embeddings pkl file (will be updated in-place)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for T5 encoding (higher = faster but more VRAM)")
    args = parser.parse_args()

    if LIBERO_PLUS_ROOT is None:
        raise RuntimeError("Set LIBERO_PLUS_ROOT to your local LIBERO-plus checkout before running this script.")
    if not TASK_MAP_FILE.exists():
        raise RuntimeError(f"Cannot find LIBERO-plus task map: {TASK_MAP_FILE}")

    pkl_path = args.pkl
    os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero-plus"))

    print(f"[1/4] Collecting all LIBERO-plus task instructions...")
    all_instructions = collect_all_instructions()
    print(f"       Found {len(all_instructions)} unique instructions across all suites.")

    print(f"[2/4] Loading existing pkl: {pkl_path}")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            cache: dict = pickle.load(f)
        print(f"       Loaded {len(cache)} existing embeddings.")
    else:
        cache = {}
        print(f"       No existing file, starting fresh.")

    missing = [inst for inst in all_instructions if inst not in cache]
    print(f"       {len(missing)} instructions need T5 computation.")

    if not missing:
        print("[4/4] All embeddings already cached. Nothing to do.")
        return

    print(f"[3/4] Computing {len(missing)} T5 embeddings (batch_size={args.batch_size})...")
    from cosmos_policy._src.predict2.inference.get_t5_emb import get_text_embedding

    t0 = time.time()
    bs = args.batch_size
    for start in range(0, len(missing), bs):
        batch = missing[start : start + bs]
        embs = get_text_embedding(batch)  # (B, 512, 1024)
        for i, inst in enumerate(batch):
            cache[inst] = embs[i : i + 1].cpu()  # keep (1, 512, 1024) shape
        done = min(start + bs, len(missing))
        elapsed = time.time() - t0
        eta = elapsed / done * (len(missing) - done) if done > 0 else 0
        print(f"       [{done}/{len(missing)}] {elapsed:.1f}s elapsed, ETA {eta:.0f}s")

    print(f"[4/4] Saving updated pkl ({len(cache)} total embeddings)...")
    if os.path.exists(pkl_path):
        backup = pkl_path + ".backup"
        shutil.copy2(pkl_path, backup)
        print(f"       Backup: {backup}")

    save_data = {}
    for k, v in cache.items():
        save_data[k] = v.cpu() if isinstance(v, torch.Tensor) else v
    with open(pkl_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"       Saved to {pkl_path}")
    print("Done.")


if __name__ == "__main__":
    main()
