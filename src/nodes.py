import torch

from .lib.sharpfin.functional import scale
from .lib.sharpfin.util import ResizeKernel


class SharpfinResizer:
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
                    {"default": "Magic Kernel Sharp 2021"},
                ),
                "srgb_conversion": (["enable", "disable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("resized_image", "out_width", "out_height")
    FUNCTION = "resize_image"
    CATEGORY = "Image/Upscaling"

    def resize_image(self, image, width, height, kernel, srgb_conversion):
        # Preserve aspect ratio - use target dimensions if ratios match, otherwise scale proportionally
        src_w, src_h = image.shape[2], image.shape[1]
        src_ratio = src_w / src_h
        target_ratio = width / height

        if target_ratio < src_ratio:
            # Target is "taller" than source - width is limiting
            width = width
            height = round(width / src_ratio)
        elif target_ratio > src_ratio:
            # Target is "wider" than source - height is limiting
            height = height
            width = round(height * src_ratio)
        # else: ratios equal, use target dimensions as-is

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

        # Validate that Nearest interpolation does not attempt to upscale
        if resize_kernel == ResizeKernel.NEAREST:
            if width > image.shape[-1] or height > image.shape[-2]:
                raise ValueError(
                    "Nearest interpolation does not support upsampling. "
                    "Please use a smoother interpolation (Bilinear, Mitchell, etc.) "
                    "or ensure the target dimensions are not larger than the source image."
                )

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

        # Store dimensions before converting back to BHWC (currently in BCHW format)
        out_width = scaled_image.shape[-1]
        out_height = scaled_image.shape[-2]

        # Convert back to BHWC
        scaled_image = scaled_image.permute(0, 2, 3, 1)

        return scaled_image, out_width, out_height
