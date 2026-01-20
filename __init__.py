from .src.nodes import MagicKernelResampler

NODE_CLASS_MAPPINGS = {"MagicKernelResampler": MagicKernelResampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MagicKernelResampler": "Sharpfin Magic Kernel Resampler"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__author__ = """ComfyUI-Sharpfin"""
__email__ = "git@grgr.dev"
__version__ = "0.0.1"