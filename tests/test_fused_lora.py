"""Check that fused LoRAs match separate q/k/v/gate LoRAs"""

import sys
import types
from dataclasses import dataclass

import pytest
import torch

# Smaller dimensions with the same Krea 2 head ratios
HEADS, KVHEADS, HEAD_DIM, HIDDEN = 48, 12, 8, 384
SPLIT_SIZES = [
    HEADS * HEAD_DIM,
    KVHEADS * HEAD_DIM,
    KVHEADS * HEAD_DIM,
    HIDDEN,
]  # 384/96/96/384
TOTAL = sum(SPLIT_SIZES)


def _load_module():
    """Import LoRA helpers with stubbed dependencies"""
    for name in ("comfy", "comfy.lora", "comfy.model_patcher", "comfy.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["comfy.lora"].model_lora_keys_unet = lambda m: {}
    sys.modules["comfy.model_patcher"].ModelPatcher = object
    sys.modules["comfy.utils"].get_attr = lambda o, k: None
    pkg = types.ModuleType("_k2")
    pkg.__path__ = []
    sys.modules.setdefault("_k2", pkg)
    models = types.ModuleType("_k2.models_krea2")
    models.QKV_GATE_ORDER = ("wq", "wk", "wv", "gate")
    sys.modules["_k2.models_krea2"] = models

    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "model_patcher" / "krea2.py"
    src = path.read_text(encoding="utf-8").replace(
        "from ..models.krea2 import QKV_GATE_ORDER",
        "from _k2.models_krea2 import QKV_GATE_ORDER",
    )
    spec = importlib.util.spec_from_loader("_k2.patcher", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = str(path)
    exec(compile(src, str(path), "exec"), mod.__dict__)  # noqa: S102
    return mod


@dataclass
class FakeAdapter:
    """Stands in for comfy.weight_adapter.lora.LoRAAdapter"""

    up: torch.Tensor
    down: torch.Tensor
    alpha: float | None = None

    @property
    def weights(self):
        return (self.up, self.down, self.alpha, None, None, None)


def fused(mod, sub):
    """Build a fused branch from projection-specific LoRAs"""
    parts = [
        ([(s, a, 1.0, None, None) for s, a in sub.get(name, [])], size)
        for name, size in zip(("wq", "wk", "wv", "gate"), SPLIT_SIZES)
    ]
    return mod.lora_branch(parts, torch.device("cpu"), torch.float32)


def make_lora(out_features, rank, gen):
    down = torch.randn(rank, HIDDEN, generator=gen, dtype=torch.float32)
    up = torch.randn(out_features, rank, generator=gen, dtype=torch.float32)
    return up, down


def check_subset(mod, present, strengths, alphas, rank=4):
    gen = torch.Generator().manual_seed(hash(present) % (2**31))
    names = ("wq", "wk", "wv", "gate")
    sub, expected = {}, {}

    for name, size in zip(names, SPLIT_SIZES):
        if name not in present:
            continue
        up, down = make_lora(size, rank, gen)
        alpha = alphas.get(name)
        strength = strengths.get(name, 1.0)
        sub[name] = [(strength, FakeAdapter(up, down, alpha))]
        scale = strength * (alpha / rank) if alpha is not None else strength
        expected[name] = up @ (down * scale)  # Separate projection contribution

    down_cat, up_cat = fused(mod, sub)
    if not present:
        assert down_cat is None and up_cat is None, "empty patch set must yield nothing"
        return "no-op OK"

    fused_delta = up_cat @ down_cat
    assert fused_delta.shape == (TOTAL, HIDDEN), f"bad shape {fused_delta.shape}"

    off, worst = 0, 0.0
    for name, size in zip(names, SPLIT_SIZES):
        slice_ = fused_delta[off : off + size]
        want = expected.get(name, torch.zeros(size, HIDDEN))
        err = (slice_ - want).abs().max().item()
        worst = max(worst, err)
        assert err < 1e-4, f"{name} slice wrong (max err {err:.2e})"
        off += size
    return f"max err {worst:.2e}"


CASES = [
    ((), {}, {}),
    (("wq",), {}, {}),
    (("gate",), {}, {}),
    (("wk", "wv"), {}, {}),
    (("wq", "wk", "wv", "gate"), {}, {}),
    (("wq", "gate"), {"wq": 0.7, "gate": -0.3}, {}),
    (("wq", "wk", "wv", "gate"), {"wq": 0.5}, {"wq": 8.0, "wv": 2.0}),
]


@pytest.mark.parametrize(
    "present,strengths,alphas",
    CASES,
    ids=lambda v: ",".join(v) or "none" if isinstance(v, tuple) else None,
)
def test_fused_matches_unfused(present, strengths, alphas):
    """Each sub-LoRA must land in its own slice of the uneven 4-way fused projection"""
    check_subset(_load_module(), present, strengths, alphas)


def test_two_loras_stack_on_same_projection():
    mod = _load_module()
    gen = torch.Generator().manual_seed(7)
    up1, down1 = make_lora(SPLIT_SIZES[0], 3, gen)
    up2, down2 = make_lora(SPLIT_SIZES[0], 5, gen)
    sub = {"wq": [(1.0, FakeAdapter(up1, down1)), (0.5, FakeAdapter(up2, down2))]}
    d, u = fused(mod, sub)
    got = (u @ d)[: SPLIT_SIZES[0]]
    want = up1 @ down1 + up2 @ (down2 * 0.5)
    assert (got - want).abs().max().item() < 1e-4
    assert d.shape[0] == 8, f"expected rank 3+5=8, got {d.shape[0]}"


def main():
    mod = _load_module()
    for present, strengths, alphas in CASES:
        label = ",".join(present) or "(none)"
        print(f"  {label:<26} {check_subset(mod, present, strengths, alphas)}")

    # Check stacked LoRAs
    gen = torch.Generator().manual_seed(7)
    up1, down1 = make_lora(SPLIT_SIZES[0], 3, gen)
    up2, down2 = make_lora(SPLIT_SIZES[0], 5, gen)
    sub = {"wq": [(1.0, FakeAdapter(up1, down1)), (0.5, FakeAdapter(up2, down2))]}
    d, u = fused(mod, sub)
    got = (u @ d)[: SPLIT_SIZES[0]]
    want = up1 @ down1 + up2 @ (down2 * 0.5)
    err = (got - want).abs().max().item()
    assert err < 1e-4, f"stacking wrong: {err:.2e}"
    assert d.shape[0] == 8, f"expected rank 3+5=8, got {d.shape[0]}"
    print(f"  {'2 LoRAs stacked on wq':<26} max err {err:.2e}, rank {d.shape[0]}")
    print("\nALL PASS")


if __name__ == "__main__":
    main()
