"""Rename checkpoint keys from diffusers names to ComfyUI's Krea 2 names"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Transformer block keys
_BLOCK_LEAF = {
    "attn.to_qkv_gate": "attn.to_qkv_gate",
    "attn.to_out.0": "attn.wo",
    "attn.to_out": "attn.wo",
    "attn.norm_q.weight": "attn.qknorm.qnorm.scale",
    "attn.norm_k.weight": "attn.qknorm.knorm.scale",
    "ff.gate": "mlp.gate",
    "ff.up": "mlp.up",
    "ff.down": "mlp.down",
    "norm1.weight": "prenorm.scale",
    "norm2.weight": "postnorm.scale",
    # Flatten the modulation table
    "scale_shift_table": "mod.lin",
}

# Keys outside transformer blocks
_TOP_LEVEL = {
    "img_in": "first",
    "time_embed.linear_1": "tmlp.0",
    "time_embed.linear_2": "tmlp.2",
    "time_mod_proj": "tproj.1",
    "txt_in.linear_1": "txtmlp.1",
    "txt_in.linear_2": "txtmlp.3",
    "txt_in.norm.weight": "txtmlp.0.scale",
    "text_fusion.projector": "txtfusion.projector",
    "final_layer.linear": "last.linear",
    "final_layer.norm.weight": "last.norm.scale",
    "final_layer.scale_shift_table": "last.modulation.lin",
}

# Text-fusion keys
_TXT_LEAF = {
    "attn.to_q": "attn.wq",
    "attn.to_k": "attn.wk",
    "attn.to_v": "attn.wv",
    "attn.to_gate": "attn.gate",
    "attn.to_out.0": "attn.wo",
    "attn.to_out": "attn.wo",
    "attn.norm_q.weight": "attn.qknorm.qnorm.scale",
    "attn.norm_k.weight": "attn.qknorm.knorm.scale",
    "ff.gate": "mlp.gate",
    "ff.up": "mlp.up",
    "ff.down": "mlp.down",
    "norm1.weight": "prenorm.scale",
    "norm2.weight": "postnorm.scale",
}


def _rename(key: str) -> str | None:
    """Map one diffusers key to its Comfy equivalent, or None if unrecognized"""
    if key.startswith("transformer_blocks."):
        rest = key[len("transformer_blocks.") :]
        idx, _, tail = rest.partition(".")
        return _rename_leaf(f"blocks.{idx}", tail, _BLOCK_LEAF)

    for kind in ("layerwise_blocks", "refiner_blocks"):
        prefix = f"text_fusion.{kind}."
        if key.startswith(prefix):
            rest = key[len(prefix) :]
            idx, _, tail = rest.partition(".")
            return _rename_leaf(f"txtfusion.{kind}.{idx}", tail, _TXT_LEAF)

    for src in sorted(_TOP_LEVEL, key=len, reverse=True):
        if key == src or key.startswith(src + "."):
            return _TOP_LEVEL[src] + key[len(src) :]

    return None


def _rename_leaf(prefix: str, tail: str, table: dict[str, str]) -> str | None:
    """Longest-prefix match tail against table and rebuild the key under prefix"""
    for src in sorted(table, key=len, reverse=True):
        if tail == src or tail.startswith(src + "."):
            return f"{prefix}.{table[src]}{tail[len(src) :]}"
    return None


def _reshape_for_comfy(comfy_key: str, value):
    """Flatten the per-block modulation table, which ComfyUI stores as one vector"""
    if comfy_key.endswith("mod.lin") and getattr(value, "ndim", 0) == 2:
        return value.reshape(-1)
    return value


def remap_state_dict(sd: dict) -> dict:
    """Return sd with ComfyUI key names; keys it doesn't recognise pass through unchanged"""
    out, renamed, passthrough = {}, 0, 0
    for k, v in sd.items():
        new_k = _rename(k)
        if new_k is None:
            out[k] = v
            passthrough += 1
        else:
            out[new_k] = _reshape_for_comfy(new_k, v)
            renamed += 1
    logger.info(
        "Nunchaku Krea 2: remapped %d keys, passed through %d", renamed, passthrough
    )
    return out
