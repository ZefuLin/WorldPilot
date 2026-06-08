import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))
import os
CHECKPOINT_BASEDIR = os.getenv('CHECKPOINT_BASEDIR', None)
DEFAULT_VGGT_PATH = os.getenv("WORLDPILOT_VGGT_PATH", "facebook/VGGT-1B")
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from WorldPilot.utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from WorldPilot.model.framework.base_framework import baseframework
from WorldPilot.model.modules.vlm import get_vlm_model
from WorldPilot.model.modules.action_model.AML_ActionHeader import get_action_model, FlowmatchingActionHead
from WorldPilot.model.tools import FRAMEWORK_REGISTRY

import random
from WorldPilot.model.modules.vggt_tools import preprocess_images, CrossAttention
from WorldPilot.model.modules.cosmos_fusion import CosmosImageFuser, CosmosActionProjector


def resize_images(images, target_size=(224, 224)):
    if isinstance(images, Image.Image):
        return images.resize(target_size)
    if isinstance(images, list):
        return [resize_images(img, target_size) for img in images]
    raise ValueError("Unsupported image type or structure.")

@FRAMEWORK_REGISTRY.register("ABot_M0")
class ABot_M0(baseframework):

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        framework_cfg = getattr(self.config, "framework", None)
        raw_use_vggt = framework_cfg.get("use_vggt", True) if framework_cfg is not None else True
        self.use_vggt = raw_use_vggt.lower() != "false" if isinstance(raw_use_vggt, str) else bool(raw_use_vggt)
        configured_vggt_path = framework_cfg.get("vggt_path", None) if framework_cfg is not None else None
        self.vggt_path = configured_vggt_path or DEFAULT_VGGT_PATH
        hidden_size = self.qwen_vl_interface.model.config.hidden_size
        if self.use_vggt:
            try:
                from vggt.models.vggt import VGGT
            except ImportError as exc:
                raise ImportError(
                    "VGGT is required when `framework.use_vggt: true`. "
                    "Install the VGGT package and set `framework.vggt_path` or `WORLDPILOT_VGGT_PATH`."
                ) from exc
            logger.info("Loading VGGT from `%s`", self.vggt_path)
            self.spatial_model = VGGT.from_pretrained(self.vggt_path)
            self.spatial_projector = nn.Linear(2048, hidden_size)
            self.fuser = CrossAttention(d_model=hidden_size, d_hidden=hidden_size, kv_dim=hidden_size)
        else:
            self.spatial_model = None
            self.spatial_projector = None
            self.fuser = None

        raw_use_cosmos = framework_cfg.get("use_cosmos", False) if framework_cfg is not None else False
        self.use_cosmos = raw_use_cosmos.lower() != "false" if isinstance(raw_use_cosmos, str) else bool(raw_use_cosmos)
        if self.use_cosmos:
            cosmos_cfg = framework_cfg.get("cosmos", {})
            raw_use_action_hint = cosmos_cfg.get("use_action_hint", True)
            self.cosmos_use_action_hint = (
                raw_use_action_hint.lower() != "false"
                if isinstance(raw_use_action_hint, str)
                else bool(raw_use_action_hint)
            )
            self.cosmos_drop_prob = float(cosmos_cfg.get("drop_prob", 0.3))
            cosmos_latent_ch = int(cosmos_cfg.get("cosmos_latent_channels", 16))
            cosmos_latent_sp = int(cosmos_cfg.get("cosmos_latent_spatial", 28))
            cosmos_chunk = int(cosmos_cfg.get("cosmos_chunk_size", 16))
            cosmos_action_dim = int(cosmos_cfg.get("cosmos_action_dim", cosmos_cfg.get("action_dim", 7)))

            self.cosmos_host = str(cosmos_cfg.get("server_host", "127.0.0.1"))
            self.cosmos_port = int(cosmos_cfg.get("server_port", 8002))
            self.cosmos_cache_dir = cosmos_cfg.get("cosmos_cache_dir")
            # Lazily initialize the Cosmos client only for evaluation or deployment inference; training always reads from cache.
            self.cosmos_client = None

            dit_input_dim = self.action_model.input_embedding_dim
            abot_horizon = self.future_action_window_size + 1

            self.cosmos_image_fuser = CosmosImageFuser(
                hidden_size=hidden_size,
                cosmos_latent_channels=cosmos_latent_ch,
                cosmos_latent_spatial=cosmos_latent_sp,
            )
            if self.cosmos_use_action_hint:
                self.cosmos_action_projector = CosmosActionProjector(
                    action_dim=cosmos_action_dim,
                    cosmos_chunk_size=cosmos_chunk,
                    abot_action_horizon=abot_horizon,
                    output_dim=dit_input_dim,
                )
            else:
                self.cosmos_action_projector = None
        else:
            self.cosmos_client = None
            self.cosmos_host = None
            self.cosmos_port = None
            self.cosmos_cache_dir = None
            self.cosmos_image_fuser = None
            self.cosmos_action_projector = None
            self.cosmos_use_action_hint = False
            self.cosmos_drop_prob = 0.0

        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        action_mask = [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None  # [B, action_dim]
        
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

            if self.use_vggt:
                # feed forward pass of vggt
                with torch.no_grad():
                    spatial_input = preprocess_images(batch_images, batch_images[0][0].size[0]).to(qwen_inputs['pixel_values'].device)
                    aggregated_tokens_list, ps_idx = self.spatial_model.aggregator(spatial_input)
                spatial_tokens = aggregated_tokens_list[-1][:, 0, ps_idx:, :]
                spatial_tokens = self.spatial_projector(spatial_tokens)
                last_hidden = self.fuser(last_hidden, spatial_tokens)

        cosmos_action_hint = None
        if self.use_cosmos:
            drop_cosmos = False
            if self.training:
                local_drop = random.random() < self.cosmos_drop_prob
                if dist.is_available() and dist.is_initialized():
                    drop_tensor = torch.tensor(
                        [int(local_drop)], device=last_hidden.device, dtype=torch.int32
                    )
                    dist.broadcast(drop_tensor, src=0)
                    drop_cosmos = bool(drop_tensor.item())
                else:
                    drop_cosmos = local_drop
            if not drop_cosmos:
                future_latents, cosmos_actions = self._get_cosmos_hints(
                    examples,
                    batch_images,
                    instructions,
                    hidden_dtype=last_hidden.dtype,
                    device=last_hidden.device,
                    source="cache",
                )
                if future_latents is not None:
                    last_hidden, cosmos_action_hint = self._apply_cosmos_hints(
                        last_hidden,
                        future_latents,
                        cosmos_actions,
                    )

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )

            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            cosmos_hint_repeated = None
            if cosmos_action_hint is not None:
                cosmos_hint_repeated = cosmos_action_hint.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_mask_repeated = None
            if action_mask is not None:
                action_mask_tensor = torch.tensor(
                    np.array(action_mask), device=last_hidden.device, dtype=torch.bool
                )  # [B, action_dim]
                action_mask_repeated = action_mask_tensor.repeat(repeated_diffusion_steps, 1)  # [B*repeated_diffusion_steps, action_dim]

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated,
                action_mask=action_mask_repeated,
                cosmos_action_hint=cosmos_hint_repeated,
            )

        return {"action_loss": action_loss}

    def _encode_cached_cosmos(self, examples: List[dict], hidden_dtype: torch.dtype, device: torch.device):
        future_latents = torch.tensor(
            np.stack([ex["cosmos_future_latents"] for ex in examples]),
            device=device,
            dtype=torch.float32,
        )
        cosmos_actions = None
        if self.cosmos_use_action_hint:
            cosmos_actions = torch.tensor(
                np.stack([ex["cosmos_action_chunk"] for ex in examples]),
                device=device,
                dtype=torch.float32,
            )
        return future_latents.to(hidden_dtype), cosmos_actions

    def _query_cosmos_online(self, batch_images, instructions, examples, hidden_dtype: torch.dtype, device: torch.device):
        if not self.use_cosmos:
            raise RuntimeError("cosmos_source='server' requires framework.use_cosmos=true.")
        if not self.cosmos_host or self.cosmos_port is None:
            raise RuntimeError(
                "cosmos_source='server' requires framework.cosmos.server_host and framework.cosmos.server_port to be configured."
            )
        if self.cosmos_client is None:
            from cosmos_bridge.cosmos_client import CosmosClient
            self.cosmos_client = CosmosClient(host=self.cosmos_host, port=self.cosmos_port)
        proprio_list = [ex.get("proprio") for ex in examples] if "proprio" in examples[0] else None
        cosmos_images = [ex.get("cosmos_image") for ex in examples]
        if cosmos_images[0] is not None:
            cosmos_query_images = cosmos_images
        else:
            cosmos_query_images = batch_images  # fallback (no cosmos_image key)
        cosmos_out = self.cosmos_client.query_batch(
            cosmos_query_images, instructions, proprio_list=proprio_list, flip_images=False
        )
        future_latents = torch.tensor(
            cosmos_out["future_image_latents"],
            device=device,
            dtype=torch.float32,
        )
        cosmos_actions = None
        if self.cosmos_use_action_hint:
            cosmos_actions = torch.tensor(
                cosmos_out["action_chunk"],
                device=device,
                dtype=torch.float32,
            )
        return future_latents.to(hidden_dtype), cosmos_actions

    def _get_cosmos_hints(self, examples, batch_images, instructions, hidden_dtype: torch.dtype, device: torch.device, source: str):
        if source == "cache":
            if "cosmos_future_latents" not in examples[0] or (
                self.cosmos_use_action_hint and "cosmos_action_chunk" not in examples[0]
            ):
                logger.warning(
                    "Cosmos cache fields are missing in this batch; skipping cosmos hints and relying on DataLoader resampling."
                )
                return None, None
            return self._encode_cached_cosmos(examples, hidden_dtype=hidden_dtype, device=device)
        if source == "server":
            return self._query_cosmos_online(
                batch_images,
                instructions,
                examples,
                hidden_dtype=hidden_dtype,
                device=device,
            )
        raise ValueError(f"Unsupported cosmos source: {source}")

    def _apply_cosmos_hints(
        self,
        last_hidden: torch.Tensor,
        future_latents: torch.Tensor,
        cosmos_actions: torch.Tensor,
    ):
        """
        Apply Cosmos fusion and action hint projection.

        Returns:
            fused_hidden:       (B, L, H) — Cosmos-enhanced VLM hidden
            cosmos_action_hint: (B, 1, D) — single advisor token for DiT input
        """
        autocast_enabled = last_hidden.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            latents_cast = future_latents.to(last_hidden.dtype)
            fused_hidden = self.cosmos_image_fuser(last_hidden, latents_cast)
            cosmos_action_hint = None
            if self.cosmos_use_action_hint:
                cosmos_action_hint = self.cosmos_action_projector(cosmos_actions).to(fused_hidden.dtype)

        return fused_hidden, cosmos_action_hint

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        cosmos_source: str = "server",
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        if cosmos_source == "server":
            if not self.use_cosmos:
                raise RuntimeError("cosmos_source='server' requires framework.use_cosmos=true.")
            if not self.cosmos_host or self.cosmos_port is None:
                raise RuntimeError(
                    "cosmos_source='server' requires framework.cosmos.server_host and framework.cosmos.server_port to be configured."
                )
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

            if self.use_vggt:
                # feed forward pass of vggt
                with torch.no_grad():
                    spatial_input = preprocess_images(batch_images, batch_images[0][0].size[0]).to(qwen_inputs['pixel_values'].device)
                    aggregated_tokens_list, ps_idx = self.spatial_model.aggregator(spatial_input)
                spatial_tokens = aggregated_tokens_list[-1][:, 0, ps_idx:, :]
                spatial_tokens = self.spatial_projector(spatial_tokens)
                last_hidden = self.fuser(last_hidden, spatial_tokens)


        cosmos_action_hint = None
        if self.use_cosmos:
            future_latents, cosmos_actions = self._get_cosmos_hints(
                examples,
                batch_images,
                instructions,
                hidden_dtype=last_hidden.dtype,
                device=last_hidden.device,
                source=cosmos_source,
            )
            if future_latents is not None:
                last_hidden, cosmos_action_hint = self._apply_cosmos_hints(
                    last_hidden,
                    future_latents,
                    cosmos_actions,
                )

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        
        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden, state, cosmos_action_hint=cosmos_action_hint,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
