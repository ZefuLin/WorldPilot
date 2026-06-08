"""
Public dataset mixtures for the WorldPilot open-source release.

This repository only supports the LIBERO training path. The original internal
tree contains many additional mixtures that depend on private or non-released
dataset roots. Keeping those import-time builders in the public release causes
empty mixtures and noisy warnings unless the original environment variables are
recreated exactly.

The public release therefore exposes only the static LIBERO mixtures used by:
  - ABot training on LIBERO
  - ABot + Cosmos joint training on LIBERO
"""

from __future__ import annotations


_LIBERO_ALL = [
    "libero_10_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
]

_LIBERO_GOS = [
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
]

_LIBERO_GO10 = [
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
]


def _uniform_mixture(dataset_names: list[str]) -> list[tuple[str, float, str, dict]]:
    weight = round(1.0 / len(dataset_names), 8)
    return [
        (dataset_name, weight, "libero_franka", {"lerobot_version": "v2.0"})
        for dataset_name in dataset_names
    ]


DATASET_NAMED_MIXTURES = {
    "libero": _uniform_mixture(_LIBERO_ALL),
    "libero_gos": _uniform_mixture(_LIBERO_GOS),
    "libero_go10": _uniform_mixture(_LIBERO_GO10),
}
