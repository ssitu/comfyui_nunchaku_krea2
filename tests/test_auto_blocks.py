"""Test loader helpers and automatic block counts without a GPU"""

import sys
import types

import pytest
import torch
from torch import nn


@pytest.fixture
def nodes_mod(monkeypatch):
    """Import the loader with stubbed dependencies"""
    free = {"bytes": 8 * 2**30}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    mm = types.ModuleType("comfy.model_management")
    mm.get_free_memory = lambda device=None: free["bytes"]
    mm.get_torch_device = lambda: torch.device("cuda", 0)
    mm.unet_offload_device = lambda: torch.device("cpu")

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []  # Allow submodule imports
    comfy.model_management = mm
    utils = types.ModuleType("comfy.utils")
    utils.load_torch_file = lambda *a, **k: ({}, {})
    utils.get_attr = lambda obj, name: obj
    comfy.utils = utils

    # Patcher dependencies
    lora = types.ModuleType("comfy.lora")
    lora.model_lora_keys_unet = lambda *a, **k: {}
    patcher = types.ModuleType("comfy.model_patcher")

    class _ModelPatcher:
        def __init__(self, *a, **k):
            pass

    patcher.ModelPatcher = _ModelPatcher

    # Attention helpers
    flux_math = types.ModuleType("comfy.ldm.flux.math")
    flux_math.apply_rope = lambda q, k, freqs: (q, k)
    attention = types.ModuleType("comfy.ldm.modules.attention")
    attention.optimized_attention_masked = lambda *a, **k: None

    stubs = {
        "comfy": comfy,
        "comfy.model_management": mm,
        "comfy.utils": utils,
        "comfy.lora": lora,
        "comfy.model_patcher": patcher,
        "comfy.ldm.flux.math": flux_math,
        "comfy.ldm.modules.attention": attention,
    }
    # Parent packages for the stubbed modules
    for pkg in ("comfy.ldm", "comfy.ldm.flux", "comfy.ldm.modules"):
        mod = types.ModuleType(pkg)
        mod.__path__ = []
        stubs[pkg] = mod

    # Folder helpers
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda kind: []
    folder_paths.get_full_path_or_raise = lambda kind, name: name
    stubs["folder_paths"] = folder_paths

    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)

    for name in list(sys.modules):
        if name.startswith("comfyui_nunchaku_krea2"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    from comfyui_nunchaku_krea2.nodes import krea2

    krea2._test_free = free
    return krea2


def _fake_model(per_block_bytes: int, n_blocks: int = 28, on_gpu: int = 15):
    """A stand-in diffusion model whose blocks have a known byte size"""
    floats = per_block_bytes // 4

    dm = nn.Module()
    dm.blocks = nn.ModuleList()
    for _ in range(n_blocks):
        b = nn.Module()
        b.w = nn.Parameter(
            torch.zeros(floats, dtype=torch.float32), requires_grad=False
        )
        dm.blocks.append(b)

    manager = types.SimpleNamespace(
        num_blocks_on_gpu=on_gpu, device=torch.device("cuda", 0)
    )
    dm._nunchaku_offload = manager
    return dm


def test_reserve_scales_with_tokens(nodes_mod):
    """Doubling tokens doubles the activation reserve"""
    one_mp = nodes_mod.activation_reserve(4608)
    two_mp = nodes_mod.activation_reserve(9216)
    assert two_mp == pytest.approx(2 * one_mp)

    # Expected reserve for 4608 tokens
    assert 1.1 * 2**30 < one_mp < 1.3 * 2**30


def test_more_tokens_means_fewer_blocks(nodes_mod):
    """Larger images keep fewer blocks on the GPU"""
    dm = _fake_model(per_block_bytes=226 * 2**20, on_gpu=15)
    nodes_mod._test_free["bytes"] = int(1.5 * 2**30)

    at_1mp = nodes_mod.blocks_for_tokens(dm, 4608)
    at_2mp = nodes_mod.blocks_for_tokens(dm, 9216)
    assert at_2mp < at_1mp, f"2 MP kept {at_2mp} blocks, 1 MP kept {at_1mp}"


def test_never_returns_zero_blocks(nodes_mod):
    """Keep at least one block on the GPU"""
    dm = _fake_model(per_block_bytes=226 * 2**20)
    nodes_mod._test_free["bytes"] = 64 * 2**20
    assert nodes_mod.blocks_for_tokens(dm, 65536) == 1


def test_never_exceeds_block_count(nodes_mod):
    """A huge GPU must not ask for more blocks than the model has"""
    dm = _fake_model(per_block_bytes=226 * 2**20, n_blocks=28)
    nodes_mod._test_free["bytes"] = 80 * 2**30
    assert nodes_mod.blocks_for_tokens(dm, 4608) == 28


def test_sizing_does_not_iterate_offloaded_blocks(nodes_mod):
    class IndexedBlocks(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self._blocks = blocks

        def __len__(self):
            return len(self._blocks)

        def __getitem__(self, idx):
            return self._blocks[idx]

        def __iter__(self):
            raise AssertionError("iteration streams weights to GPU")

    dm = _fake_model(per_block_bytes=1024, n_blocks=3, on_gpu=1)
    dm.blocks = IndexedBlocks(dm.blocks)
    assert nodes_mod.blocks_for_tokens(dm, 4608) == 3


def test_retune_repairs_placement_before_measuring(nodes_mod, monkeypatch):
    from comfyui_nunchaku_krea2.models import krea2

    dm = _fake_model(per_block_bytes=1024, n_blocks=3, on_gpu=2)
    calls = []
    dm._nunchaku_offload.set_device = lambda device, force=False: calls.append(
        (device, force)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def free_memory(device=None):
        assert calls == [(torch.device("cuda", 0), True)]
        return 8 * 2**30

    monkeypatch.setattr(
        nodes_mod.comfy.model_management, "get_free_memory", free_memory
    )
    krea2._retune_blocks_for_tokens(dm, (torch.empty(1, 16, 128, 128),), {})
    # Reuse the estimate for the same resolution
    krea2._retune_blocks_for_tokens(dm, (torch.empty(1, 16, 128, 128),), {})
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mode,free,expected",
    [
        ("auto", 100, [300]),
        ("enable", 100, [300]),
        ("disable", 100, []),
        ("auto", 300, []),
    ],
)
def test_reclaim_cached_memory_before_sizing(
    nodes_mod, monkeypatch, mode, free, expected
):
    mm = nodes_mod.comfy.model_management
    calls = []
    monkeypatch.setattr(
        mm, "free_memory", lambda needed, device: calls.append(needed), raising=False
    )
    nodes_mod._test_free["bytes"] = free
    model = nn.Linear(25, 1, bias=False)  # 100 bytes
    nodes_mod._prepare_offload_memory(mode, model, torch.device("cuda", 0), reserve=200)
    assert calls == expected


def test_reclaim_skips_cpu(nodes_mod, monkeypatch):
    mm = nodes_mod.comfy.model_management
    monkeypatch.setattr(
        mm,
        "free_memory",
        lambda *a: pytest.fail("must not evict CPU models"),
        raising=False,
    )
    nodes_mod._prepare_offload_memory(
        "auto", nn.Linear(1, 1), torch.device("cpu"), reserve=200
    )


@pytest.fixture
def scale_model(nodes_mod, monkeypatch):
    class Linear(nn.Module):
        def __init__(self, precision="nvfp4"):
            super().__init__()
            self.wtscale = 1.0 if precision == "nvfp4" else None
            self.wcscales = (
                nn.Parameter(torch.ones(4, device="meta", dtype=torch.bfloat16))
                if precision == "nvfp4"
                else None
            )

    linear_mod = types.ModuleType("nunchaku.models.linear")
    linear_mod.SVDQW4A4Linear = Linear
    monkeypatch.setitem(sys.modules, linear_mod.__name__, linear_mod)
    model = nn.Module()
    model.blocks = nn.ModuleList([Linear(), Linear()])
    model.int4 = Linear("int4")
    return model


@pytest.mark.parametrize("scale", [torch.tensor(2.5), 2.5])
def test_nvfp4_scales_are_loaded(nodes_mod, scale_model, scale):
    channel_scale = torch.full((4,), 3.0, dtype=torch.bfloat16)
    sd = {
        "blocks.0.wtscale": scale,
        "blocks.1.wtscale": torch.tensor(0.5),
        "blocks.0.wcscales": channel_scale,
    }
    nodes_mod._load_nvfp4_scales(scale_model, sd)
    assert scale_model.blocks[0].wtscale == 2.5
    assert isinstance(scale_model.blocks[0].wtscale, float)
    assert scale_model.blocks[1].wtscale == 0.5
    assert not any(key.endswith(".wtscale") for key in sd)
    assert sd["blocks.0.wcscales"] is channel_scale
    missing = sd["blocks.1.wcscales"]
    assert missing.device.type == "cpu" and missing.dtype == torch.bfloat16
    torch.testing.assert_close(missing, torch.ones_like(missing))
    result = scale_model.load_state_dict(sd, strict=False, assign=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert all(not p.is_meta for p in scale_model.parameters())


def test_nvfp4_missing_scales_use_defaults(nodes_mod, scale_model):
    scale_model.blocks[0].wtscale = 7.0
    sd = {}
    nodes_mod._load_nvfp4_scales(scale_model, sd)
    assert all(block.wtscale == 1.0 for block in scale_model.blocks)
    assert set(sd) == {"blocks.0.wcscales", "blocks.1.wcscales"}
    assert scale_model.int4.wtscale is None


def test_int4_scale_loading_is_unchanged(nodes_mod, scale_model):
    sd = {"proj_up": torch.ones(4, 4)}
    before = dict(sd)
    nodes_mod._load_nvfp4_scales(scale_model.int4, sd)
    assert sd == before
    assert scale_model.int4.wtscale is None


def test_nvfp4_global_scale_must_be_scalar(nodes_mod, scale_model):
    with pytest.raises(RuntimeError, match="Scalar"):
        nodes_mod._load_nvfp4_scales(scale_model, {"blocks.0.wtscale": torch.ones(2)})


def test_loader_prepares_scales_before_weights(nodes_mod, scale_model, monkeypatch):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = nn.Module()

        def load_model_weights(self, sd, prefix, assign=False):
            assert assign and self.diffusion_model.blocks[0].wtscale == 2.5
            result = self.diffusion_model.load_state_dict(
                sd, strict=False, assign=assign
            )
            assert not result.missing_keys and not result.unexpected_keys

    def patch_model(dm, **kwargs):
        dm.blocks = scale_model.blocks
        dm.int4 = scale_model.int4

    model = Model()
    metadata = {"quantization_config": '{"precision":"nvfp4"}'}
    monkeypatch.setattr(
        nodes_mod.comfy.utils,
        "load_torch_file",
        lambda *a, **k: ({"blocks.0.wtscale": torch.tensor(2.5)}, metadata),
    )
    utils = types.ModuleType("nunchaku.utils")
    utils.check_hardware_compatibility = lambda *a: None
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setattr(
        nodes_mod,
        "_build_model_config",
        lambda *a: types.SimpleNamespace(get_model=lambda *a: model),
    )
    monkeypatch.setattr(nodes_mod, "patch_model", patch_model)
    nodes_mod.NunchakuKrea2DiTLoader().load_model("checkpoint", cpu_offload="disable")
    assert all(not p.is_meta for p in model.parameters())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp16_conversion_matches_model_dtype(nodes_mod, dtype):
    model = nn.Linear(3, 2, bias=False, dtype=dtype, device="meta")
    model.register_buffer("scale", torch.empty(2, dtype=dtype, device="meta"))
    source = torch.tensor(
        [[1.0, float("nan"), float("inf")], [-float("inf"), 1e10, -1e10]],
        dtype=torch.bfloat16,
    )
    sd = {"weight": source, "scale": torch.ones(2, dtype=torch.bfloat16)}
    nodes_mod._convert_fp16_weights(model, sd)
    if dtype == torch.float16:
        assert sd["weight"].dtype == torch.float16
        assert torch.isfinite(sd["weight"]).all()
        torch.testing.assert_close(
            sd["weight"],
            torch.tensor([[1, 0, 65504], [-65504, 65504, -65504]], dtype=dtype),
        )
        assert sd["scale"].dtype == torch.float16
    else:
        assert sd["weight"] is source
    model.load_state_dict(sd, assign=True)
    assert model.weight.dtype == model.scale.dtype == dtype
    if dtype == torch.float16:
        assert model(torch.zeros(1, 3, dtype=dtype)).dtype == dtype


def test_fp16_conversion_leaves_other_tensors_unchanged(nodes_mod):
    model = nn.Module()
    for name, dtype in (
        ("qweight", torch.int8),
        ("wscales", torch.float8_e4m3fn),
        ("bf16", torch.bfloat16),
        ("fp16", torch.float16),
        ("missing", torch.float16),
    ):
        model.register_parameter(
            name,
            nn.Parameter(
                torch.empty(2, dtype=dtype, device="meta"), requires_grad=False
            ),
        )
    sd = {
        "qweight": torch.zeros(2, dtype=torch.int8),
        "wscales": torch.ones(2, dtype=torch.float8_e4m3fn),
        "bf16": torch.ones(2, dtype=torch.bfloat16),
        "fp16": torch.ones(2, dtype=torch.float16),
        "extra": torch.ones(2, dtype=torch.bfloat16),
    }
    before = dict(sd)
    nodes_mod._convert_fp16_weights(model, sd)
    assert sd.keys() == before.keys()
    assert all(sd[key] is value for key, value in before.items())


def test_fp16_conversion_skips_lora_placeholders(nodes_mod):
    model = nn.Module()
    model.register_parameter(
        "proj_up", nn.Parameter(torch.empty(2, 2, dtype=torch.float16, device="meta"))
    )

    def hook(module, state, prefix, metadata):
        state[prefix + "weight"] = torch.empty(0, dtype=torch.float16)
        return state

    model._register_state_dict_hook(hook)
    extra = torch.ones(2, 2, dtype=torch.bfloat16)
    sd = {"proj_up": torch.ones(2, 2, dtype=torch.bfloat16), "weight": extra}
    nodes_mod._convert_fp16_weights(model, sd)
    assert sd["proj_up"].dtype == torch.float16
    assert sd["weight"] is extra


def test_loader_converts_fp16_before_weights(nodes_mod, monkeypatch):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = nn.Module()
            self.diffusion_model.first = nn.Linear(
                2, 2, bias=False, dtype=torch.float16
            )

        def load_model_weights(self, sd, prefix, assign=False):
            assert assign and sd["first.weight"].dtype == torch.float16
            self.diffusion_model.load_state_dict(sd, assign=assign)

    model = Model()
    monkeypatch.setattr(
        nodes_mod.comfy.utils,
        "load_torch_file",
        lambda *a, **k: (
            {"first.weight": torch.ones(2, 2, dtype=torch.bfloat16)},
            {"quantization_config": '{"precision":"int4"}'},
        ),
    )
    utils = types.ModuleType("nunchaku.utils")
    utils.check_hardware_compatibility = lambda *a: None
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setattr(
        nodes_mod,
        "_build_model_config",
        lambda *a: types.SimpleNamespace(get_model=lambda *a: model),
    )
    monkeypatch.setattr(nodes_mod, "patch_model", lambda *a, **k: None)
    monkeypatch.setattr(nodes_mod, "_load_nvfp4_scales", lambda *a: None)
    nodes_mod.NunchakuKrea2DiTLoader().load_model("checkpoint", cpu_offload="disable")
    assert (
        model.diffusion_model.first(torch.ones(1, 2, dtype=torch.float16)).dtype
        == torch.float16
    )
