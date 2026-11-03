"""Test LoRA key discovery after quantization without a GPU"""

import sys
import types

import pytest
import torch
from torch import nn

HIDDEN, HEADS, KVHEADS, HEADDIM = 64, 4, 2, 16


class FakeSVDQLinear(nn.Module):
    """Quantized test layer with qweight instead of weight"""

    def __init__(self, in_features, out_features, rank=8, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory = {
            "device": kwargs.get("device"),
            "dtype": kwargs.get("torch_dtype", torch.float32),
        }
        self.qweight = nn.Parameter(
            torch.zeros(
                out_features,
                in_features // 2,
                dtype=torch.uint8,
                device=factory["device"],
            ),
            requires_grad=False,
        )
        self.proj_down = nn.Parameter(
            torch.zeros(rank, in_features, **factory), requires_grad=False
        )
        self.proj_up = nn.Parameter(
            torch.zeros(out_features, rank, **factory), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, **factory))
            if kwargs.get("bias")
            else None
        )

    @classmethod
    def from_linear(cls, linear, **kwargs):
        return cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            torch_dtype=linear.weight.dtype,
            device=linear.weight.device,
            **kwargs,
        )


@pytest.fixture
def krea2_mod(monkeypatch):
    """Import models.krea2 with nunchaku and comfy stubbed out"""
    linear_mod = types.ModuleType("nunchaku.models.linear")
    linear_mod.SVDQW4A4Linear = FakeSVDQLinear
    models_mod = types.ModuleType("nunchaku.models")
    models_mod.linear = linear_mod

    flux_math = types.ModuleType("comfy.ldm.flux.math")
    flux_math.apply_rope = lambda q, k, freqs: (q, k)
    attention = types.ModuleType("comfy.ldm.modules.attention")
    attention.optimized_attention_masked = lambda *a, **k: None

    stubs = {
        "nunchaku": types.ModuleType("nunchaku"),
        "nunchaku.models": models_mod,
        "nunchaku.models.linear": linear_mod,
        "comfy.ldm.flux.math": flux_math,
        "comfy.ldm.modules.attention": attention,
    }
    for pkg in ("comfy", "comfy.ldm", "comfy.ldm.flux", "comfy.ldm.modules"):
        mod = types.ModuleType(pkg)
        mod.__path__ = []
        stubs[pkg] = mod

    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    for name in list(sys.modules):
        if name.startswith("comfyui_nunchaku_krea2"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    from comfyui_nunchaku_krea2.models import krea2

    return krea2


def _fake_dit(n_blocks=2):
    """Minimal stand-in for comfy's SingleStreamDiT: blocks with attn + mlp linears"""

    def attn():
        m = nn.Module()
        m.heads, m.kvheads, m.headdim = HEADS, KVHEADS, HEADDIM
        m.qknorm = nn.Module()
        m.wq = nn.Linear(HIDDEN, HEADS * HEADDIM, bias=False)
        m.wk = nn.Linear(HIDDEN, KVHEADS * HEADDIM, bias=False)
        m.wv = nn.Linear(HIDDEN, KVHEADS * HEADDIM, bias=False)
        m.gate = nn.Linear(HIDDEN, HIDDEN, bias=False)
        m.wo = nn.Linear(HEADS * HEADDIM, HIDDEN, bias=False)
        return m

    def mlp():
        m = nn.Module()
        m.gate = nn.Linear(HIDDEN, HIDDEN * 2, bias=False)
        m.up = nn.Linear(HIDDEN, HIDDEN * 2, bias=False)
        m.down = nn.Linear(HIDDEN * 2, HIDDEN, bias=False)
        return m

    dit = nn.Module()
    dit.blocks = nn.ModuleList()
    for _ in range(n_blocks):
        b = nn.Module()
        b.attn = attn()
        b.mlp = mlp()
        dit.blocks.append(b)
    return dit


# Original projection names
PROJECTIONS = [
    "attn.wq",
    "attn.wk",
    "attn.wv",
    "attn.gate",
    "attn.wo",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
]


def test_lora_key_names_survive_quantization(krea2_mod):
    """Every projection name present before patching must still be advertised after"""
    dit = _fake_dit()
    before = {k[: -len(".weight")] for k in dit.state_dict() if k.endswith(".weight")}

    krea2_mod.patch_model(dit, precision="int4", rank=8)
    after = {k[: -len(".weight")] for k in dit.state_dict() if k.endswith(".weight")}

    missing = sorted(before - after)
    assert not missing, (
        f"quantization hid {len(missing)} LoRA key names, e.g. {missing[:4]}"
    )

    for i in range(len(dit.blocks)):
        for proj in PROJECTIONS:
            assert f"blocks.{i}.{proj}" in after, f"blocks.{i}.{proj} not advertised"


def test_fused_projection_is_not_advertised_raw(krea2_mod):
    """Advertise original projections, not the fused weight"""
    dit = _fake_dit()
    krea2_mod.patch_model(dit, precision="int4", rank=8)
    keys = set(dit.state_dict())
    assert "blocks.0.attn.to_qkv_gate.weight" not in keys
    # Keep the quantized weight
    assert "blocks.0.attn.to_qkv_gate.qweight" in keys


def test_advertised_entries_are_metadata_only(krea2_mod):
    """The advertised entries must not carry weights that could be mistaken for real data"""
    dit = _fake_dit()
    krea2_mod.patch_model(dit, precision="int4", rank=8)
    sd = dit.state_dict()
    for proj in PROJECTIONS:
        assert sd[f"blocks.0.{proj}.weight"].numel() == 0


def test_offload_container_keeps_original_lora_paths(krea2_mod):
    dit = _fake_dit()
    krea2_mod.patch_model(dit, precision="int4", rank=8)
    before = set(dit.state_dict())
    dit.blocks = krea2_mod._OffloadedBlocks(dit.blocks, object())
    assert set(dit.state_dict()) == before
    assert "blocks.0.attn.wq.weight" in dit.state_dict()
    assert repr(dit.blocks) == "_OffloadedBlocks(2 blocks)"


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_replacements_are_shape_only_then_assignable(krea2_mod, lazy, bias):
    linear = nn.Linear(64, 128, bias=bias, dtype=torch.bfloat16)
    if lazy:
        linear.weight = None
        linear.bias = None
        linear.weight_comfy_model_dtype = torch.bfloat16
        linear.comfy_need_lazy_init_bias = bias
    quantized = krea2_mod._to_svdq(FakeSVDQLinear, linear, rank=16, precision="int4")
    assert all(p.is_meta for p in quantized.parameters())
    assert quantized.proj_down.dtype == torch.bfloat16
    assert quantized.proj_down.shape == (16, 64)
    assert (quantized.bias is not None) == bias
    checkpoint = {
        key: torch.zeros_like(tensor, device="cpu")
        for key, tensor in quantized.state_dict().items()
    }
    quantized.load_state_dict(checkpoint, assign=True)
    assert all(not p.is_meta for p in quantized.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_swiglu_preserves_inputs_and_outputs(krea2_mod, dtype):
    mlp = krea2_mod.NunchakuKrea2SwiGLU.__new__(krea2_mod.NunchakuKrea2SwiGLU)
    nn.Module.__init__(mlp)
    # Projections share the input; leave it unchanged
    mlp.gate = mlp.up = mlp.down = nn.Identity()
    x = torch.randn(2, 16, 64, dtype=dtype)
    before = x.clone()
    with torch.inference_mode():
        expected = torch.nn.functional.silu(x) * x
        actual = mlp(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
