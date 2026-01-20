from .src.nodes import SharpfinResizer

NODE_CLASS_MAPPINGS = {"SharpfinResizer": SharpfinResizer}
NODE_DISPLAY_NAME_MAPPINGS = {"SharpfinResizer": "Sharpfin Magic Image Resize"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__author__ = """ComfyUI-Sharpfin"""
__email__ = "git@grgr.dev"
__version__ = "1.0.0"
