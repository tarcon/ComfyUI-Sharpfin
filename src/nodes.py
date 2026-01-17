import torch

from sharpfin.functional import scale
from sharpfin.util import ResizeKernel


class MagicKernelResampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "kernel": (
                    [
                        "Nearest",
                        "Bilinear",
                        "Mitchell",
                        "Catmull-Rom",
                        "B-Spline",
                        "Lanczos2",
                        "Lanczos3",
                        "Magic Kernel",
                        "Magic Kernel Sharp 2013",
                        "Magic Kernel Sharp 2021",
                    ],
                ),
                "srgb_conversion": (["enable", "disable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "Sharpfin/Image"

    def resize_image(self, image, width, height, kernel, srgb_conversion):
        # Convert to BCHW
        image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW

        # Map kernel string to ResizeKernel enum value
        kernel_mapping = {
            "Nearest": ResizeKernel.NEAREST,
            "Bilinear": ResizeKernel.BILINEAR,
            "Mitchell": ResizeKernel.MITCHELL,
            "Catmull-Rom": ResizeKernel.CATMULL_ROM,
            "B-Spline": ResizeKernel.B_SPLINE,
            "Lanczos2": ResizeKernel.LANCZOS2,
            "Lanczos3": ResizeKernel.LANCZOS3,
            "Magic Kernel": ResizeKernel.MAGIC_KERNEL,
            "Magic Kernel Sharp 2013": ResizeKernel.MAGIC_KERNEL_SHARP_2013,
            "Magic Kernel Sharp 2021": ResizeKernel.MAGIC_KERNEL_SHARP_2021,
        }
        resize_kernel = kernel_mapping.get(kernel)
        if resize_kernel is None:
            raise ValueError(f"Unknown kernel: {kernel}")

        # Handle srgb conversion
        do_srgb_conversion = srgb_conversion == "enable"

        # Determine device and dtype for scale function
        device = image.device
        dtype = torch.float32

        # Call scale function with out_res=(height, width)
        scaled_image = scale(
            image,
            (int(height), int(width)),
            resize_kernel=resize_kernel,
            device=device,
            dtype=dtype,
            do_srgb_conversion=do_srgb_conversion,
        )

        # Convert back to BHWC
        scaled_image = scaled_image.permute(0, 2, 3, 1)
        return (scaled_image,)


NODE_CLASS_MAPPINGS = {"MagicKernelResampler": MagicKernelResampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MagicKernelResampler": "Sharpfin Magic Kernel Resampler"}
