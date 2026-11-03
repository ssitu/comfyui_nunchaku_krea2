"""Apply ComfyUI LoRAs to Krea 2 SVDQuant layers"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from comfy.model_patcher import ModelPatcher
from comfy.utils import get_attr
from torch import nn

from ..models.krea2 import QKV_GATE_ORDER


def _module_bytes(module, device=None):
    return sum(
        t.numel() * t.element_size()
        for t in (*module.parameters(), *module.buffers())
        if device is None or t.device == device
    )


def _get_lowrank(linear):
    from nunchaku.lora.flux.nunchaku_converter import unpack_lowrank_weight

    return unpack_lowrank_weight(linear.proj_down, down=True), unpack_lowrank_weight(
        linear.proj_up, down=False
    )


def _set_lowrank(linear, down, up):
    from nunchaku.lora.flux.nunchaku_converter import pack_lowrank_weight

    linear.proj_down = nn.Parameter(
        pack_lowrank_weight(down, down=True), requires_grad=False
    )
    linear.proj_up = nn.Parameter(
        pack_lowrank_weight(up, down=False), requires_grad=False
    )


def lora_branch(parts, device, dtype):
    """Combine (patches, size) output slices into one LoRA branch"""
    total = sum(size for _, size in parts)
    downs, ups, start = [], [], 0
    for patches, size in parts:
        for strength, adapter, *_ in patches:
            weights = getattr(adapter, "weights", ())
            if len(weights) != 6 or any(w is not None for w in weights[3:]):
                raise NotImplementedError(
                    "Only plain LoRAs are supported on quantized Krea 2 layers."
                )
            up, down, alpha = weights[:3]
            rank = down.shape[0]
            if alpha is not None:
                strength = strength * (alpha / rank)
            padded_up = torch.zeros(total, rank, device=device, dtype=dtype)
            padded_up[start : start + size] = up.to(device=device, dtype=dtype)
            downs.append(down.to(device=device, dtype=dtype) * strength)
            ups.append(padded_up)
        start += size
    if not downs:
        return None, None
    return torch.cat(downs, dim=0), torch.cat(ups, dim=1)


class NunchakuKrea2ModelPatcher(ModelPatcher):
    """Applies LoRAs to the low-rank branches and restores the originals on unpatch"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.svdq_backup = {}

    def clone(self, *args, **kwargs):
        n = super().clone(*args, **kwargs)
        n.svdq_backup = self.svdq_backup
        return n

    def _manager(self):
        """The CPU offload manager, if the loader enabled offloading"""
        return getattr(
            getattr(self.model, "diffusion_model", None), "_nunchaku_offload", None
        )

    def load(
        self,
        device_to=None,
        lowvram_model_memory=0,
        force_patch_weights=False,
        full_load=False,
    ):
        """Load the whole model; ComfyUI can't partially load SVDQuant layers"""
        manager = self._manager()
        if manager is not None:
            return self._load_offloaded(
                manager, device_to or self.load_device, force_patch_weights
            )
        self._apply_loras()
        return super().load(
            device_to, lowvram_model_memory, force_patch_weights, full_load=True
        )

    def _load_offloaded(self, manager, device_to, force_patch_weights):
        """Apply patches on the CPU, then let the offload manager move blocks to the GPU"""
        with self.use_ejected():
            torch.cuda.synchronize(device_to)
            manager.buffer_blocks.clear()
            self.model.to("cpu")
            self._apply_loras()
            super().load(
                torch.device("cpu"),
                lowvram_model_memory=1e64,
                force_patch_weights=force_patch_weights,
                full_load=True,
            )
            self._pad_offload_ranks(manager)
            manager.buffer_blocks = [copy.deepcopy(manager.blocks[0]) for _ in range(2)]
            manager.current_block_idx = 0
            manager.set_device(device_to, force=True)
            self.model.device = device_to
            self.model.model_loaded_weight_memory = self.loaded_size()

    def _pad_offload_ranks(self, manager):
        """Pad every block's low-rank branch to the same rank, as the offload buffers need"""
        from nunchaku.models.linear import SVDQW4A4Linear

        names = {id(module): name for name, module in self.model.named_modules()}
        groups = {}
        for block in manager.blocks:
            for name, module in block.named_modules():
                if isinstance(module, SVDQW4A4Linear):
                    groups.setdefault(name, []).append(module)
        for modules in groups.values():
            rank = max(module.proj_up.shape[1] for module in modules)
            for module in modules:
                extra = rank - module.proj_up.shape[1]
                if extra:
                    self._backup(names[id(module)], module)
                    down, up = _get_lowrank(module)
                    _set_lowrank(
                        module, F.pad(down, (0, 0, 0, extra)), F.pad(up, (0, extra))
                    )

    def loaded_size(self):
        manager = self._manager()
        if manager is None:
            return super().loaded_size()
        device = torch.device(self.load_device)
        return _module_bytes(self.model, device) + sum(
            _module_bytes(b, device) for b in manager.buffer_blocks
        )

    def model_size(self):
        manager = self._manager()
        if manager is None:
            return super().model_size()
        return _module_bytes(self.model) + sum(
            _module_bytes(b) for b in manager.buffer_blocks
        )

    def _backup(self, name, linear):
        if name not in self.svdq_backup:
            self.svdq_backup[name] = (linear.proj_down, linear.proj_up)

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        # Ensure the first load runs
        if self.model.current_weight_patches_uuid is None:
            extra_memory = max(0, extra_memory)
        return super().partially_load(device_to, extra_memory, force_patch_weights)

    def _apply_loras(self):
        """Append LoRAs to quantized layers' low-rank branches"""
        from nunchaku.models.linear import SVDQW4A4Linear

        with self.use_ejected():
            for name, linear in self.model.named_modules():
                # Skip layers already patched
                if not isinstance(linear, SVDQW4A4Linear) or name in self.svdq_backup:
                    continue
                if name.endswith("to_qkv_gate"):
                    base = name[: -len("to_qkv_gate")]
                    attn = get_attr(self.model, name[: -len(".to_qkv_gate")])
                    parts = [
                        (self.patches.get(f"{base}{sub}.weight", []), size)
                        for sub, size in zip(QKV_GATE_ORDER, attn.split_sizes)
                    ]
                else:
                    parts = [
                        (self.patches.get(f"{name}.weight", []), linear.out_features)
                    ]
                lora_down, lora_up = lora_branch(
                    parts, linear.proj_down.device, linear.proj_down.dtype
                )
                if lora_down is None:
                    continue
                self._backup(name, linear)
                down, up = _get_lowrank(linear)
                _set_lowrank(
                    linear,
                    torch.cat([down, lora_down]),
                    torch.cat([up, lora_up], dim=1),
                )

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        # Unload whole models, not individual SVDQuant layers
        return 0

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        manager = self._manager()
        if unpatch_weights and manager is not None:
            torch.cuda.synchronize(manager.device)
            manager.buffer_blocks.clear()
        super().unpatch_model(device_to, unpatch_weights)
        if not unpatch_weights:
            return
        for name, (proj_down, proj_up) in self.svdq_backup.items():
            linear = get_attr(self.model, name)
            linear.proj_down = nn.Parameter(
                proj_down.to(device=device_to), requires_grad=False
            )
            linear.proj_up = nn.Parameter(
                proj_up.to(device=device_to), requires_grad=False
            )
        self.svdq_backup.clear()
