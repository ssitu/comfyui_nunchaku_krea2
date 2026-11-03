"""Test LoRA application, restoration and offload padding with stubbed dependencies"""

import pytest
import torch
from torch import nn

CIN, COUT, RANK, LORA_RANK = 64, 128, 32, 8


class FakeSVDQLinear(nn.Module):
    """Stand-in for a Nunchaku quantized layer"""

    def __init__(self, cin=CIN, cout=COUT, rank=RANK):
        super().__init__()
        self.proj_down = nn.Parameter(torch.randn(rank, cin), requires_grad=False)
        self.proj_up = nn.Parameter(torch.randn(cout, rank), requires_grad=False)
        self.out_features = cout

    @property
    def rank(self):
        return self.proj_down.shape[0]

    def lowrank(self, x):
        return x @ self.proj_down.T @ self.proj_up.T


@pytest.fixture
def patcher_mod(monkeypatch):
    """Import the patcher with stubbed dependencies"""
    import sys
    import types

    # Stub Nunchaku
    nunchaku = types.ModuleType("nunchaku")
    linear_mod = types.ModuleType("nunchaku.models.linear")
    linear_mod.SVDQW4A4Linear = FakeSVDQLinear
    models_mod = types.ModuleType("nunchaku.models")
    models_mod.linear = linear_mod
    conv = types.ModuleType("nunchaku.lora.flux.nunchaku_converter")
    conv.pack_lowrank_weight = lambda t, down: t
    conv.unpack_lowrank_weight = lambda t, down: t
    for name, mod in [
        ("nunchaku", nunchaku),
        ("nunchaku.models", models_mod),
        ("nunchaku.models.linear", linear_mod),
        ("nunchaku.lora", types.ModuleType("nunchaku.lora")),
        ("nunchaku.lora.flux", types.ModuleType("nunchaku.lora.flux")),
        ("nunchaku.lora.flux.nunchaku_converter", conv),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    # Stub ComfyUI
    class _ModelPatcher:
        def __init__(
            self,
            model,
            load_device,
            offload_device,
            size=0,
            weight_inplace_update=False,
        ):
            self.model = model
            self.load_device = load_device
            self.offload_device = offload_device
            self.patches = {}
            self.backup = {}
            self.patches_uuid = None

        def use_ejected(self):
            import contextlib

            return contextlib.nullcontext()

        def unpatch_model(self, device_to=None, unpatch_weights=True):
            pass

        def _load_list(self, for_dynamic=False, default_device=None):
            return []

        def clone(self):
            n = type(self)(self.model, self.load_device, self.offload_device)
            n.patches = self.patches
            return n

    lora_mod = types.ModuleType("comfy.lora")
    # LoRA targets used in these tests
    lora_mod.model_lora_keys_unet = lambda model: {
        "x": "blocks.0.attn.wq.weight",
        "y": "blocks.0.attn.wk.weight",
        "z": "blocks.0.attn.wv.weight",
        "w": "blocks.0.attn.gate.weight",
        "v": "blocks.0.ff.linear.weight",
    }
    mp_mod = types.ModuleType("comfy.model_patcher")
    mp_mod.ModelPatcher = _ModelPatcher
    utils_mod = types.ModuleType("comfy.utils")

    def _get_attr(obj, attr):
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj

    utils_mod.get_attr = _get_attr

    # Allow ComfyUI submodule imports
    comfy_mod = types.ModuleType("comfy")
    comfy_mod.__path__ = []

    def _pkg(name):
        m = types.ModuleType(name)
        m.__path__ = []
        return m

    # Stub attention helpers
    math_mod = types.ModuleType("comfy.ldm.flux.math")
    math_mod.apply_rope = lambda xq, xk, freqs_cis: (xq, xk)
    attn_mod = types.ModuleType("comfy.ldm.modules.attention")
    attn_mod.optimized_attention_masked = lambda q, k, v, heads, mask=None, **kw: q

    for name, mod in [
        ("comfy", comfy_mod),
        ("comfy.lora", lora_mod),
        ("comfy.model_patcher", mp_mod),
        ("comfy.utils", utils_mod),
        ("comfy.ldm", _pkg("comfy.ldm")),
        ("comfy.ldm.flux", _pkg("comfy.ldm.flux")),
        ("comfy.ldm.flux.math", math_mod),
        ("comfy.ldm.modules", _pkg("comfy.ldm.modules")),
        ("comfy.ldm.modules.attention", attn_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    sys.modules.pop("comfyui_nunchaku_krea2.model_patcher.krea2", None)
    import importlib

    return importlib.import_module("comfyui_nunchaku_krea2.model_patcher.krea2")


class FakeAdapter:
    """Stand-in for a ComfyUI LoRA adapter"""

    def __init__(self, up, down, alpha=None):
        self.weights = (up, down, alpha, None, None, None)


def _make_model():
    model = nn.Module()
    blocks = nn.ModuleList([nn.Module()])
    blocks[0].ff = nn.Module()
    blocks[0].ff.linear = FakeSVDQLinear()
    model.blocks = blocks
    return model


def test_double_apply_does_not_grow_rank(patcher_mod):
    """Applying a LoRA twice must not change its rank or effect"""
    model = _make_model()
    linear = model.blocks[0].ff.linear

    p = patcher_mod.NunchakuKrea2ModelPatcher(
        model, load_device="cpu", offload_device="cpu"
    )
    up = torch.randn(COUT, LORA_RANK) * 0.02
    down = torch.randn(LORA_RANK, CIN) * 0.02
    p.patches["blocks.0.ff.linear.weight"] = [
        (1.0, FakeAdapter(up, down), 1.0, None, None)
    ]

    x = torch.randn(4, CIN)
    base = linear.lowrank(x).clone()

    p._apply_loras()
    rank_once, out_once = linear.rank, linear.lowrank(x).clone()
    assert rank_once == RANK + LORA_RANK

    p._apply_loras()
    assert linear.rank == rank_once, (
        f"rank grew {rank_once} -> {linear.rank}: LoRA applied twice"
    )
    torch.testing.assert_close(linear.lowrank(x), out_once)

    # Check the LoRA contribution
    expected = base + x @ down.T @ up.T
    torch.testing.assert_close(out_once, expected, rtol=1e-4, atol=1e-5)


def test_unpatch_restores_and_allows_reapply(patcher_mod):
    model = _make_model()
    linear = model.blocks[0].ff.linear

    p = patcher_mod.NunchakuKrea2ModelPatcher(
        model, load_device="cpu", offload_device="cpu"
    )
    up = torch.randn(COUT, LORA_RANK) * 0.02
    down = torch.randn(LORA_RANK, CIN) * 0.02
    p.patches["blocks.0.ff.linear.weight"] = [
        (1.0, FakeAdapter(up, down), 1.0, None, None)
    ]

    x = torch.randn(4, CIN)
    pristine = linear.lowrank(x).clone()

    p._apply_loras()
    patched = linear.lowrank(x).clone()
    p.unpatch_model()

    assert linear.rank == RANK, "unpatch must restore the original rank"
    torch.testing.assert_close(linear.lowrank(x), pristine)

    # Reapplying restores the same patched result
    p.patches["blocks.0.ff.linear.weight"] = [
        (1.0, FakeAdapter(up, down), 1.0, None, None)
    ]
    p._apply_loras()
    assert linear.rank == RANK + LORA_RANK
    torch.testing.assert_close(linear.lowrank(x), patched)


def test_unpatch_to_other_device_restores_parameters(patcher_mod):
    model = _make_model()
    p = patcher_mod.NunchakuKrea2ModelPatcher(model, "cpu", "cpu")
    linear = model.blocks[0].ff.linear
    p._backup("blocks.0.ff.linear", linear)
    p.unpatch_model(torch.device("meta"))
    assert isinstance(linear.proj_down, nn.Parameter) and linear.proj_down.is_meta
    assert isinstance(linear.proj_up, nn.Parameter) and linear.proj_up.is_meta


def test_sparse_lora_padding_preserves_maths_and_unpatch(patcher_mod):
    import types

    model = _make_model()
    other = nn.Module()
    other.ff = nn.Module()
    other.ff.linear = FakeSVDQLinear()
    model.blocks.append(other)
    patcher = patcher_mod.NunchakuKrea2ModelPatcher(model, "cpu", "cpu")
    patcher.patches["blocks.0.ff.linear.weight"] = [
        (
            0.5,
            FakeAdapter(torch.randn(COUT, LORA_RANK), torch.randn(LORA_RANK, CIN)),
            1,
            None,
            None,
        )
    ]
    x = torch.randn(2, CIN)
    before = other.ff.linear.lowrank(x).clone()
    patcher._apply_loras()
    first = model.blocks[0].ff.linear.lowrank(x).clone()
    patcher._pad_offload_ranks(types.SimpleNamespace(blocks=list(model.blocks)))
    assert other.ff.linear.rank == model.blocks[0].ff.linear.rank == RANK + LORA_RANK
    torch.testing.assert_close(other.ff.linear.lowrank(x), before)
    torch.testing.assert_close(model.blocks[0].ff.linear.lowrank(x), first)
    patcher.unpatch_model()
    assert other.ff.linear.rank == model.blocks[0].ff.linear.rank == RANK


@pytest.mark.parametrize("kind", ["dora", "mid", "reshape", "other"])
def test_unsupported_quantized_adapter_fails_clearly(patcher_mod, kind):
    model = _make_model()
    patcher = patcher_mod.NunchakuKrea2ModelPatcher(model, "cpu", "cpu")
    adapter = FakeAdapter(torch.ones(COUT, LORA_RANK), torch.ones(LORA_RANK, CIN))
    if kind == "other":
        adapter = object()
    else:
        values = list(adapter.weights)
        values[{"mid": 3, "dora": 4, "reshape": 5}[kind]] = torch.ones(1)
        adapter.weights = tuple(values)
    patcher.patches["blocks.0.ff.linear.weight"] = [(1.0, adapter, 1.0, None, None)]
    with pytest.raises(NotImplementedError):
        patcher._apply_loras()


def test_first_managed_load_cannot_skip_negative_budget(patcher_mod, monkeypatch):
    base = patcher_mod.NunchakuKrea2ModelPatcher.__mro__[1]
    calls = []
    monkeypatch.setattr(
        base, "partially_load", lambda self, *args: calls.append(args), raising=False
    )
    model = _make_model()
    model.current_weight_patches_uuid = None
    patcher = patcher_mod.NunchakuKrea2ModelPatcher(model, "cpu", "cpu")
    patcher.partially_load("cpu", -1024)
    assert calls[-1][1] == 0
    model.current_weight_patches_uuid = "already-loaded"
    patcher.partially_load("cpu", -1024)
    assert calls[-1][1] == -1024


def test_managed_load_applies_patches_before_staging(patcher_mod, monkeypatch):
    import types

    base = patcher_mod.NunchakuKrea2ModelPatcher.__mro__[1]
    order = []
    root = nn.Module()
    root.diffusion_model = _make_model()
    root.current_weight_patches_uuid = None
    manager = types.SimpleNamespace(
        blocks=list(root.diffusion_model.blocks), buffer_blocks=[]
    )
    manager.set_device = lambda device, force=False: order.append("stage")
    root.diffusion_model._nunchaku_offload = manager
    patcher = patcher_mod.NunchakuKrea2ModelPatcher(
        root, torch.device("cpu"), torch.device("cpu")
    )
    patcher.patches["diffusion_model.blocks.0.ff.linear.weight"] = [
        (
            1,
            FakeAdapter(torch.ones(COUT, LORA_RANK), torch.ones(LORA_RANK, CIN)),
            1,
            None,
            None,
        )
    ]

    def cpu_load(self, device_to, **kwargs):
        assert device_to.type == "cpu"
        assert root.diffusion_model.blocks[0].ff.linear.rank == RANK + LORA_RANK
        self.model.current_weight_patches_uuid = self.patches_uuid
        order.append("patch")

    monkeypatch.setattr(base, "load", cpu_load, raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    patcher.load(torch.device("cpu"))
    assert order == ["patch", "stage"]
    assert len(manager.buffer_blocks) == 2
    assert root.diffusion_model.blocks[0].ff.linear.rank == RANK + LORA_RANK


def test_partially_unload_is_a_no_op(patcher_mod):
    """ComfyUI must unload the whole model instead of moving some SVDQuant layers"""
    p = patcher_mod.NunchakuKrea2ModelPatcher(_make_model(), "cpu", "cpu")
    assert p.partially_unload("cpu", 1234) == 0
