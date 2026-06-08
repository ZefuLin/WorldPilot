"""
Lightweight Cosmos WebSocket client for ABot integration.

Zero cosmos dependency — only uses ABot's existing WebsocketClientPolicy + msgpack_numpy.
Supports synchronous queries and asynchronous prefetch for training.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


class CosmosClient:
    """
    WebSocket client that talks to cosmos_bridge.cosmos_server.

    The server handles all Cosmos-specific preprocessing (flipud, JPEG
    compression, centro crop, proprio normalization).  This client just
    sends raw numpy images and receives structured predictions.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8002):
        from deployment.model_server.tools.websocket_policy_client import (
            WebsocketClientPolicy,
        )

        self._client = WebsocketClientPolicy(host=host, port=port)
        self._prefetch_result: Optional[Dict[str, np.ndarray]] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        log.info("CosmosClient connected to ws://%s:%d", host, port)

    def query_single(
        self,
        primary_img: np.ndarray,
        wrist_img: np.ndarray,
        lang: str,
        proprio: Optional[np.ndarray] = None,
        flip_images: bool = False,
        right_wrist_img: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Synchronous single-sample query.

        Args:
            primary_img: (H, W, 3) uint8 — raw image.
            wrist_img:   (H, W, 3) uint8 — raw wrist image.
            right_wrist_img: Optional secondary wrist image for ALOHA/RoboTwin.
            lang:        Language instruction string.
            proprio:     (9,) float32, optional raw proprio.
            flip_images: Whether server should flipud before Cosmos inference.
                         False for training (DataLoader images are normal orientation).
                         True  for LIBERO eval (env images are OpenGL y-up, upside-down).

        Returns:
            dict with keys: action_chunk, future_image_latents,
            future_primary, future_wrist, value.
        """
        msg = {
            "type": "infer",
            "primary": np.ascontiguousarray(primary_img),
            "wrist": np.ascontiguousarray(wrist_img),
            "left_wrist": np.ascontiguousarray(wrist_img),
            "right_wrist": np.ascontiguousarray(right_wrist_img if right_wrist_img is not None else wrist_img),
            "lang": lang,
            "proprio": proprio,
            "flip_images": flip_images,
        }
        response = self._client.predict_action(msg)
        return response["data"]

    def query_batch(
        self,
        images_list: List[list],
        instructions: List[str],
        proprio_list: Optional[List[np.ndarray]] = None,
        flip_images: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Batch query — sends samples one by one over the persistent connection.

        Args:
            images_list:  List of [primary_PIL, wrist_PIL, ...] per sample.
            instructions: List of language instruction strings.
            proprio_list: Optional list of (9,) float32 proprio arrays.
            flip_images:  Whether server should flipud (see query_single).

        Returns:
            dict of stacked numpy arrays (batch dim = 0).
        """
        results = []
        for i, (imgs, inst) in enumerate(zip(images_list, instructions)):
            primary = np.asarray(imgs[0])
            wrist = np.asarray(imgs[1]) if len(imgs) > 1 else np.asarray(imgs[0])
            right_wrist = np.asarray(imgs[2]) if len(imgs) > 2 else None
            proprio = proprio_list[i] if proprio_list else None
            results.append(
                self.query_single(
                    primary,
                    wrist,
                    inst,
                    proprio,
                    flip_images=flip_images,
                    right_wrist_img=right_wrist,
                )
            )
        return _stack_batch(results)

    def prefetch_async(
        self,
        images_list: List[list],
        instructions: List[str],
        proprio_list: Optional[List[np.ndarray]] = None,
        flip_images: bool = False,
    ) -> None:
        """Start a background thread to prefetch the next batch's Cosmos predictions."""
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join()

        def _worker():
            try:
                self._prefetch_result = self.query_batch(
                    images_list, instructions, proprio_list, flip_images
                )
            except Exception:
                log.exception("Cosmos prefetch failed")
                self._prefetch_result = None

        self._prefetch_thread = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()

    def get_prefetch_result(self) -> Optional[Dict[str, np.ndarray]]:
        """Block until the prefetch thread finishes, then return the result."""
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None
        result = self._prefetch_result
        self._prefetch_result = None
        return result

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


def _stack_batch(results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    batch = {
        "future_image_latents": np.stack([r["future_image_latents"] for r in results]),
        "action_chunk": np.stack([r["action_chunk"] for r in results]),
        "value": np.array([r["value"] for r in results], dtype=np.float32),
        "future_primary": np.stack([r["future_primary"] for r in results]),
        "future_wrist": np.stack([r["future_wrist"] for r in results]),
    }
    if any("future_wrist2" in r for r in results):
        batch["future_wrist2"] = np.stack([r.get("future_wrist2", r["future_wrist"]) for r in results])
    return batch
