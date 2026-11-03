"""Nunchaku W4A4 layers for ComfyUI's Krea 2 model"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked
from einops import rearrange
from torch import nn

logger = logging.getLogger(__name__)

# Fused projection order
QKV_GATE_ORDER = ("wq", "wk", "wv", "gate")


def _linear_dtype(linear: nn.Module) -> torch.dtype:
    """Dtype of a linear's weight, even when ComfyUI hasn't allocated the weight yet"""
    weight = getattr(linear, "weight", None)
    if weight is not None:
        return weight.dtype
    return getattr(linear, "weight_comfy_model_dtype", None) or torch.bfloat16


def _to_svdq(svdq_cls, linear: nn.Module, **kwargs):
    """Create a W4A4 layer without allocating its weights"""
    kwargs.setdefault("device", "meta")
    has_bias = getattr(
        linear, "comfy_need_lazy_init_bias", getattr(linear, "bias", None) is not None
    )
    return svdq_cls(
        in_features=kwargs.pop("in_features", linear.in_features),
        out_features=linear.out_features,
        bias=bool(has_bias),
        torch_dtype=kwargs.pop("torch_dtype", _linear_dtype(linear)),
        **kwargs,
    )


def qkv_gate_split_sizes(attn: nn.Module) -> list[int]:
    """Output sizes of q, k, v and gate in the fused projection (6144, 1536, 1536, 6144)"""
    return [
        attn.heads * attn.headdim,  # q
        attn.kvheads * attn.headdim,  # k
        attn.kvheads * attn.headdim,  # v
        attn.gate.out_features,  # gate (hidden_size)
    ]


class NunchakuKrea2Attention(nn.Module):
    """Krea 2 attention with q/k/v/gate in one W4A4 linear; otherwise same as ComfyUI's"""

    def __init__(self, orig_attn: nn.Module, **kwargs):
        super().__init__()
        from nunchaku.models.linear import SVDQW4A4Linear

        self.heads = orig_attn.heads
        self.kvheads = orig_attn.kvheads
        self.headdim = orig_attn.headdim
        self.qknorm = orig_attn.qknorm
        self.split_sizes = qkv_gate_split_sizes(orig_attn)

        # The checkpoint supplies these weights
        srcs = [orig_attn.wq, orig_attn.wk, orig_attn.wv, orig_attn.gate]
        assert all(lin.in_features == srcs[0].in_features for lin in srcs)
        with torch.device("meta"):
            fused = nn.Linear(
                srcs[0].in_features,
                sum(lin.out_features for lin in srcs),
                bias=all(lin.bias is not None for lin in srcs),
                dtype=_linear_dtype(srcs[0]),
            )
        self.to_qkv_gate = _to_svdq(SVDQW4A4Linear, fused, **kwargs)
        self.wo = _to_svdq(SVDQW4A4Linear, orig_attn.wo, **kwargs)

    def forward(self, x, freqs=None, mask=None, transformer_options=None):
        if transformer_options is None:
            transformer_options = {}
        transformer_patches = transformer_options.get("patches", {})
        extra_options = transformer_options.copy()

        q, k, v, gate = torch.split(self.to_qkv_gate(x), self.split_sizes, dim=-1)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        q, k = self.qknorm(q, k)

        if (
            "block_index" in transformer_options
            and "attn1_patch" in transformer_patches
        ):
            for p in transformer_patches["attn1_patch"]:
                out = p(q, k, v, pe=freqs, attn_mask=mask, extra_options=extra_options)
                q, k, v = out.get("q", q), out.get("k", k), out.get("v", v)
                freqs, mask = out.get("pe", freqs), out.get("attn_mask", mask)

        if freqs is not None:
            q, k = apply_rope(q, k, freqs)

        # Expand k/v heads to keep masked attention efficient
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        out = optimized_attention_masked(
            q,
            k,
            v,
            self.heads,
            mask=mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )

        if (
            "block_index" in transformer_options
            and "attn1_output_patch" in transformer_patches
        ):
            for p in transformer_patches["attn1_output_patch"]:
                out = p(out, extra_options)

        return self.wo(out * F.sigmoid(gate))


class NunchakuKrea2SwiGLU(nn.Module):
    """Krea 2 MLP with W4A4 gate, up and down linears"""

    def __init__(self, orig_mlp: nn.Module, **kwargs):
        super().__init__()
        from nunchaku.models.linear import SVDQW4A4Linear

        self.gate = _to_svdq(SVDQW4A4Linear, orig_mlp.gate, **kwargs)
        self.up = _to_svdq(SVDQW4A4Linear, orig_mlp.up, **kwargs)
        self.down = _to_svdq(SVDQW4A4Linear, orig_mlp.down, **kwargs)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)).mul_(self.up(x)))


def _advertise_lora_keys(block: nn.Module) -> None:
    """Expose original layer names as empty entries for LoRA matching"""
    empty = torch.empty(0)
    names = [f"attn.{sub}" for sub in QKV_GATE_ORDER]
    names += ["attn.wo", "mlp.gate", "mlp.up", "mlp.down"]

    def hook(module, state, prefix, local_metadata, _names=tuple(names)):
        for name in _names:
            state[f"{prefix}{name}.weight"] = empty
        return state

    block._register_state_dict_hook(hook)


def patch_model(model: nn.Module, precision: str = "int4", rank: int = 32) -> nn.Module:
    """Replace each block's attention and MLP with W4A4 versions, in place"""
    kwargs = {"precision": precision, "rank": rank}
    n = 0
    for block in model.blocks:
        block.attn = NunchakuKrea2Attention(block.attn, **kwargs)
        block.mlp = NunchakuKrea2SwiGLU(block.mlp, **kwargs)
        _advertise_lora_keys(block)
        n += 1
    logger.info(
        "Nunchaku Krea 2: patched %d blocks (%d quantized linears), precision=%s rank=%d",
        n,
        n * 5,
        precision,
        rank,
    )
    return model


def _retune_blocks_for_tokens(model: nn.Module, args, kwargs) -> None:
    """Lower the automatic block count for larger images"""
    manager = getattr(model, "_nunchaku_offload", None)
    if manager is None or getattr(model, "_nunchaku_blocks_pinned_by_user", False):
        return

    x = kwargs.get("x", args[0] if args else None)
    if x is None or not hasattr(x, "shape") or x.ndim < 2:
        return

    from ..nodes.krea2 import blocks_for_tokens

    patch = getattr(model, "patch", 2)
    # Include frames for 5D latents
    tokens = (x.shape[-2] // patch) * (x.shape[-1] // patch)
    if x.ndim == 5:
        tokens *= x.shape[-3]
    if tokens == getattr(model, "_nunchaku_tuned_tokens", None):
        return
    model._nunchaku_tuned_tokens = tokens

    # Restore GPU placement before measuring free memory
    dev = torch.device("cuda", torch.cuda.current_device())
    p = next(model.blocks[0].parameters(), None)
    manager.set_device(dev, force=p is not None and p.device != dev)
    want = blocks_for_tokens(model, tokens)
    if want >= manager.num_blocks_on_gpu:
        return

    logger.info(
        "Nunchaku Krea 2: %d tokens needs more activation VRAM than assumed -- "
        "reducing %d -> %d blocks on GPU.",
        tokens,
        manager.num_blocks_on_gpu,
        want,
    )
    manager.num_blocks_on_gpu = want
    # Force restaging on the same device
    manager.set_device(manager.device, force=True)
    torch.cuda.empty_cache()


class _OffloadedBlocks(nn.ModuleList):
    """Block list that streams blocks to the GPU as the forward pass iterates over it"""

    def __init__(self, blocks, manager):
        super().__init__(blocks)
        self._manager = manager

    def __iter__(self):
        manager = self._manager
        stream = torch.cuda.current_stream()
        dev = torch.device("cuda", torch.cuda.current_device())
        p = next(self[0].parameters(), None)
        manager.set_device(dev, force=p is not None and p.device != dev)
        manager.initialize(stream)
        for i in range(len(self)):
            yield manager.get_block(i)
            manager.step(stream)

    def __repr__(self):
        return f"{type(self).__name__}({len(self)} blocks)"


def enable_cpu_offload(
    model: nn.Module, num_blocks_on_gpu: int = 16, use_pin_memory: bool = True
) -> None:
    """Stream blocks from RAM while keeping the other layers on the GPU"""
    from nunchaku.models.utils import CPUOffloadManager

    gpu_modules = [
        m
        for m in (
            getattr(model, name, None)
            for name in ("first", "tproj", "tmlp", "txtmlp", "txtfusion", "last")
        )
        if m is not None
    ]

    manager = CPUOffloadManager(
        list(model.blocks),
        # Match the block iterator's device index
        device=torch.device("cuda", torch.cuda.current_device()),
        use_pin_memory=use_pin_memory,
        on_gpu_modules=gpu_modules,
        num_blocks_on_gpu=num_blocks_on_gpu,
    )
    model._nunchaku_offload = manager

    # Preserve layer names for LoRA matching
    model.blocks = _OffloadedBlocks(model.blocks, manager)

    # Restore GPU modules before each forward pass
    original_forward = model.forward

    def forward_with_gpu_modules(*args, **kwargs):
        dev = torch.device("cuda", torch.cuda.current_device())
        for mod in gpu_modules:
            p = next(mod.parameters(), None)
            if p is not None and p.device != dev:
                mod.to(dev)
        _retune_blocks_for_tokens(model, args, kwargs)
        return original_forward(*args, **kwargs)

    model.forward = forward_with_gpu_modules

    logger.info(
        "Nunchaku Krea 2: CPU offload enabled (%d/%d blocks on GPU, %d GPU modules)",
        num_blocks_on_gpu,
        len(model.blocks),
        len(gpu_modules),
    )
