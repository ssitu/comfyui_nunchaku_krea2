"""ComfyUI loader node for Nunchaku SVDQuant (W4A4) Krea 2 checkpoints"""

from __future__ import annotations

import json
import logging

import comfy.model_management
import comfy.utils
import torch

from ..model_patcher.krea2 import NunchakuKrea2ModelPatcher
from ..models.krea2 import enable_cpu_offload, patch_model
from ..models.state_dict import remap_state_dict
from .utils import get_filename_list, get_full_path_or_raise

logger = logging.getLogger(__name__)


class NunchakuKrea2DiTLoader:
    """Load a Nunchaku Krea 2 checkpoint as a ComfyUI model"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    get_filename_list("diffusion_models"),
                    {
                        "tooltip": "A Nunchaku SVDQuant Krea 2 checkpoint (svdq-int4_r32-*.safetensors)."
                    },
                ),
            },
            "optional": {
                "precision": (
                    ["auto", "int4", "nvfp4"],
                    {
                        "default": "auto",
                        "tooltip": "'auto' uses the checkpoint's setting. "
                        "nvfp4 is for Blackwell GPUs; int4 for older GPUs.",
                    },
                ),
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "auto",
                        "tooltip": "Stream transformer blocks from system RAM when the model "
                        "doesn't fit in VRAM. 'auto' enables it only when needed.",
                    },
                ),
                "num_blocks_on_gpu": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 28,
                        "tooltip": "Blocks kept on the GPU when offloading (0 = automatic). "
                        "Higher is faster but uses more VRAM.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Krea 2 DiT Loader"

    def load_model(
        self,
        model_name: str,
        precision: str = "auto",
        cpu_offload: str = "auto",
        num_blocks_on_gpu: int = 0,
    ):
        from nunchaku.utils import check_hardware_compatibility

        model_path = get_full_path_or_raise("diffusion_models", model_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)

        quantization_config = _normalize_quant_config(
            json.loads((metadata or {}).get("quantization_config", "{}"))
        )
        rank = quantization_config.get("rank", 32)
        if precision == "auto":
            precision = quantization_config.get("precision") or _precision_from_gpu()
        if precision == "fp4":
            precision = "nvfp4"
        check_hardware_compatibility(
            quantization_config, comfy.model_management.get_torch_device()
        )
        logger.info(
            "Nunchaku Krea 2: precision=%s rank=%d from %s", precision, rank, model_name
        )

        sd = remap_state_dict(sd)

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        model_config = _build_model_config(sd, load_device)
        model = model_config.get_model(sd, "")
        # Move before creating meta layers
        model = model.to(offload_device)
        patch_model(model.diffusion_model, precision=precision, rank=rank)
        _load_nvfp4_scales(model.diffusion_model, sd)
        _convert_fp16_weights(model.diffusion_model, sd)
        # assign=True fills the meta layers
        model.load_model_weights(sd, "", assign=True)
        _assert_no_meta(model)

        _prepare_offload_memory(cpu_offload, model, load_device, _ACTIVATION_RESERVE)
        offload = _resolve_cpu_offload(cpu_offload, model, load_device)
        if offload:
            n_blocks = num_blocks_on_gpu or _auto_blocks_on_gpu(model, load_device)
            enable_cpu_offload(model.diffusion_model, num_blocks_on_gpu=n_blocks)
            # Preserve the user's block count
            model.diffusion_model._nunchaku_blocks_pinned_by_user = bool(
                num_blocks_on_gpu
            )

        patcher = NunchakuKrea2ModelPatcher(
            model, load_device=load_device, offload_device=offload_device
        )
        return (patcher,)


# Estimated activation bytes per token
_ACTIVATION_BYTES_PER_TOKEN = 136_500

# Load-time reserve for roughly 1024x1024 images
_ACTIVATION_RESERVE = int(1.8 * 2**30)


def activation_reserve(tokens: int) -> int:
    """VRAM to keep free for activations, with headroom for CFG and the VAE decode"""
    return 2 * tokens * _ACTIVATION_BYTES_PER_TOKEN


def _prepare_offload_memory(mode: str, model, load_device, reserve: int) -> None:
    """Let ComfyUI unload cached models before measuring free VRAM"""
    if mode == "disable" or load_device.type != "cuda":
        return
    needed = _module_bytes(model) + reserve
    if comfy.model_management.get_free_memory(load_device) < needed:
        comfy.model_management.free_memory(needed, load_device)


def _resolve_cpu_offload(mode: str, model, load_device) -> bool:
    """Decide whether to stream blocks from CPU"""
    if mode == "disable":
        return False
    if mode == "enable":
        return True

    size = model.model_size() if hasattr(model, "model_size") else _module_bytes(model)
    free = comfy.model_management.get_free_memory(load_device)
    needed = size + _ACTIVATION_RESERVE
    if free < needed:
        logger.info(
            "Nunchaku Krea 2: %.2f GiB free but %.2f GiB needed (%.2f weights + %.2f "
            "activations), enabling CPU offload.",
            free / 2**30,
            needed / 2**30,
            size / 2**30,
            _ACTIVATION_RESERVE / 2**30,
        )
        return True
    return False


def _auto_blocks_on_gpu(model, load_device, tokens: int | None = None) -> int:
    """Most blocks that fit on the GPU while leaving room for activations"""
    dm = model.diffusion_model
    blocks = dm.blocks
    per_block = _module_bytes(blocks[0])
    resident = _module_bytes(dm) - sum(
        _module_bytes(blocks[i]) for i in range(len(blocks))
    )

    reserve = _ACTIVATION_RESERVE if tokens is None else activation_reserve(tokens)
    free = comfy.model_management.get_free_memory(load_device)
    # Reserve two offload buffers
    budget = free - reserve - resident - 2 * per_block
    n = max(1, min(len(blocks), int(budget // per_block)))
    logger.info(
        "Nunchaku Krea 2: keeping %d/%d blocks on GPU (%.0f MiB each, %.2f GiB free, "
        "%.2f GiB reserved for activations%s).",
        n,
        len(blocks),
        per_block / 2**20,
        free / 2**30,
        reserve / 2**30,
        "" if tokens is None else f" at {tokens} tokens",
    )
    return n


def blocks_for_tokens(diffusion_model, tokens: int) -> int:
    """Choose a block count for the token count, including loaded blocks"""
    blocks = diffusion_model.blocks
    per_block = _module_bytes(blocks[0])
    manager = diffusion_model._nunchaku_offload

    free = comfy.model_management.get_free_memory(
        torch.device("cuda", torch.cuda.current_device())
    )
    budget = free + manager.num_blocks_on_gpu * per_block - activation_reserve(tokens)
    return max(1, min(len(blocks), int(budget // per_block)))


def _module_bytes(mod) -> int:
    return sum(p.numel() * p.element_size() for p in mod.parameters()) + sum(
        b.numel() * b.element_size() for b in mod.buffers()
    )


def _assert_no_meta(model) -> None:
    """Raise if the checkpoint left any parameter unfilled"""
    meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta:
        raise RuntimeError(
            f"Nunchaku Krea 2: {len(meta)} parameters are still on the meta device after "
            f"loading (e.g. {', '.join(meta[:3])}). The checkpoint did not supply them; "
            f"refusing to return a model that would generate garbage."
        )


def _convert_fp16_weights(model, sd: dict) -> None:
    """Convert BF16 checkpoint tensors where the model expects FP16"""
    for name, target in (*model.named_parameters(), *model.named_buffers()):
        source = sd.get(name)
        if (
            target.dtype == torch.float16
            and source is not None
            and source.dtype == torch.bfloat16
        ):
            sd[name] = torch.nan_to_num(
                source.to(torch.float16), nan=0.0, posinf=65504, neginf=-65504
            )


def _load_nvfp4_scales(model, sd: dict) -> None:
    """Load global FP4 scales and fill missing channel scales"""
    from nunchaku.models.linear import SVDQW4A4Linear

    for name, module in model.named_modules():
        if not isinstance(module, SVDQW4A4Linear) or module.wtscale is None:
            continue
        scale = sd.pop(f"{name}.wtscale", 1.0)
        module.wtscale = float(
            scale.item() if isinstance(scale, torch.Tensor) else scale
        )
        key = f"{name}.wcscales"
        if key not in sd:
            sd[key] = torch.ones_like(module.wcscales, device="cpu")


# Nunchaku weight formats
_WEIGHT_DTYPE = {"int4": "int4", "fp4": "fp4_e2m1_all", "nvfp4": "fp4_e2m1_all"}


def _normalize_quant_config(qc: dict) -> dict:
    """Accept flat and nested quantization metadata"""
    qc = dict(qc)
    if "weight" in qc:
        qc.setdefault(
            "precision",
            "nvfp4" if qc["weight"].get("dtype") == "fp4_e2m1_all" else "int4",
        )
        return qc
    precision = qc.get("precision")
    if precision:
        qc["weight"] = {
            "dtype": _WEIGHT_DTYPE.get(precision, precision),
            "group_size": qc.get("group_size", 64),
        }
    return qc


def _precision_from_gpu() -> str:
    from nunchaku.utils import get_precision

    return get_precision()


def _build_model_config(sd: dict, load_device):
    """ComfyUI's own Krea 2 model config, with the inference dtype set"""
    import comfy.model_management
    from comfy.supported_models import Krea2

    unet_config = dict(Krea2.unet_config)
    unet_config["layers"] = _count_blocks(sd)
    config = Krea2(unet_config)
    config.unet_config = unet_config

    dtype = comfy.model_management.unet_dtype(
        model_params=_param_count(sd),
        supported_dtypes=config.supported_inference_dtypes,
    )
    manual_cast = comfy.model_management.unet_manual_cast(
        dtype, load_device, config.supported_inference_dtypes
    )
    config.set_inference_dtype(dtype, manual_cast, device=load_device)
    logger.info("Nunchaku Krea 2: unet dtype=%s manual_cast=%s", dtype, manual_cast)
    return config


def _param_count(sd: dict) -> int:
    return sum(t.numel() for t in sd.values() if hasattr(t, "numel"))


def _count_blocks(sd: dict) -> int:
    idx = -1
    for k in sd:
        if k.startswith("blocks."):
            try:
                idx = max(idx, int(k.split(".")[1]))
            except (IndexError, ValueError):
                continue
    return idx + 1
