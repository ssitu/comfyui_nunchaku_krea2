"""Check key remapping using a published checkpoint's header"""

import importlib.util
import json
import pathlib
import struct
import urllib.request

URL = (
    "https://huggingface.co/felipesztutman/Krea-2-Turbo-W4A4-Nunchaku/"
    "resolve/main/svdq-int4_r32-krea-2-turbo.safetensors"
)
UA = {"User-Agent": "Mozilla/5.0"}


def load_remap():
    path = pathlib.Path(__file__).resolve().parents[1] / "models" / "state_dict.py"
    spec = importlib.util.spec_from_file_location("_k2_sd", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.remap_state_dict


def fetch_keys():
    def rng(a, b):
        req = urllib.request.Request(URL, headers={**UA, "Range": f"bytes={a}-{b}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    n = struct.unpack("<Q", rng(0, 7))[0]
    idx = json.loads(rng(8, 8 + n - 1))
    idx.pop("__metadata__", None)
    return sorted(idx)


def main():
    remap = load_remap()
    keys = fetch_keys()
    print(f"checkpoint tensors: {len(keys)}")

    out = remap({k: None for k in keys})
    assert len(out) == len(keys), f"key collision: {len(keys)} -> {len(out)}"

    # Check that all diffusers keys were renamed
    leftovers = [
        k
        for k in out
        if k.startswith(
            (
                "transformer_blocks.",
                "img_in",
                "time_embed.",
                "time_mod_proj",
                "txt_in.",
                "final_layer.",
                "text_fusion.",
            )
        )
    ]
    assert not leftovers, f"{len(leftovers)} keys not remapped, e.g. {leftovers[:5]}"

    # Check ComfyUI target names
    valid = ("blocks.", "txtfusion.", "first.", "tmlp.", "tproj.", "txtmlp.", "last.")
    unknown = [k for k in out if not k.startswith(valid)]
    assert not unknown, f"{len(unknown)} unknown targets, e.g. {unknown[:5]}"

    quantized = {k.rsplit(".", 1)[0] for k in out if k.endswith(".qweight")}
    assert len(quantized) == 140, (
        f"expected 140 quantized linears, got {len(quantized)}"
    )

    fused = {k for k in quantized if k.endswith("attn.to_qkv_gate")}
    assert len(fused) == 28, f"expected 28 fused to_qkv_gate, got {len(fused)}"

    # Text fusion stays unquantized
    assert not [k for k in quantized if k.startswith("txtfusion.")], (
        "txtfusion must not be quantized"
    )

    print(f"  remapped OK, {len(quantized)} quantized linears, {len(fused)} fused attn")
    for sample in (
        "blocks.0.attn.to_qkv_gate.qweight",
        "blocks.0.attn.wo.qweight",
        "blocks.0.mlp.down.qweight",
        "blocks.0.prenorm.scale",
        "blocks.0.postnorm.scale",
        "blocks.0.attn.qknorm.qnorm.scale",
        "blocks.0.mod.lin",
        "first.weight",
        "tmlp.0.weight",
        "tproj.1.weight",
        "last.linear.weight",
        "last.norm.scale",
        "last.modulation.lin",
        "txtfusion.projector.weight",
        "txtfusion.layerwise_blocks.0.attn.wq.weight",
        "txtfusion.refiner_blocks.1.prenorm.scale",
    ):
        assert sample in out, f"missing expected key {sample}"
        print(f"  present: {sample}")
    print("\nALL PASS")


if __name__ == "__main__":
    main()
