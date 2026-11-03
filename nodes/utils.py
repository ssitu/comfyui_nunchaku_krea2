"""Folder helpers for ComfyUI and ComfyUI-LTS"""

try:  # ComfyUI-LTS
    from comfy.cmd import folder_paths
    from comfy.model_downloader import get_filename_list, get_full_path_or_raise
except (ImportError, ModuleNotFoundError):  # ComfyUI
    import folder_paths
    from folder_paths import get_filename_list, get_full_path_or_raise

__all__ = ["folder_paths", "get_filename_list", "get_full_path_or_raise"]
