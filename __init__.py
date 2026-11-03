"""Nunchaku SVDQuant loader for Krea 2"""

import logging

NODE_CLASS_MAPPINGS = {}
try:
    from .nodes.krea2 import NunchakuKrea2DiTLoader

    NODE_CLASS_MAPPINGS["NunchakuKrea2DiTLoader"] = NunchakuKrea2DiTLoader
except ImportError:
    logging.getLogger(__name__).exception("NunchakuKrea2DiTLoader import failed")
NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
