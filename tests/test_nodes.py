import pytest
import torch

from src.nodes import SharpfinResizer


def create_test_image():
    """Create a dummy image tensor of shape [1, 64, 64, 3] with values in range [0, 1]."""
    return torch.ones(1, 64, 64, 3) * 0.5  # All pixels are 0.5


@pytest.fixture
def sut():
    return SharpfinResizer()


def test_initialization(sut):
    """Test that the node initializes correctly."""
    assert isinstance(sut, SharpfinResizer)


def test_input_types():
    """Check INPUT_TYPES structure and required fields."""
    input_types = SharpfinResizer.INPUT_TYPES()
    assert "required" in input_types
    required_fields = ["image", "width", "height", "kernel", "srgb_conversion"]
    for field in required_fields:
        assert field in input_types["required"]


def test_resize_with_default_params(sut):
    """Test resizing with default parameters."""
    img = create_test_image()
    result = sut.resize_image(img, 128, 128, "Magic Kernel", "enable")

    # Check output shape
    assert result[0].shape == (1, 128, 128, 3)
    # Ensure values are within [0,1]
    assert torch.all((result[0] >= 0) & (result[0] <= 1))


def test_srgb_conversion(sut):
    """Test that sRGB conversion affects the result."""
    # Use a gradient image with varying values to show sRGB conversion difference
    img = torch.zeros(1, 64, 64, 3)
    for i in range(64):
        img[0, i, :, :] = i / 63.0  # Gradient from 0 to 1

    enabled_result = sut.resize_image(img, 32, 32, "Bilinear", "enable")
    disabled_result = sut.resize_image(img, 32, 32, "Bilinear", "disable")

    # Check results differ when srgb conversion is enabled vs disabled
    assert not torch.allclose(enabled_result[0], disabled_result[0], atol=1e-5)


def test_kernel_mapping(sut):
    """Test that different kernel strings map to correct ResizeKernel enums."""
    kernels = [
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
    ]

    for kernel in kernels:
        try:
            # Test each kernel runs without error
            img = create_test_image()
            sut.resize_image(img, 64, 64, kernel, "enable")
            assert True
        except Exception as e:
            pytest.fail(f"Kernel '{kernel}' raised exception: {e}")


def test_edge_cases(sut):
    """Test edge cases for width and height."""
    img = create_test_image()

    # Test downscaling to small size
    result_min = sut.resize_image(img, 32, 32, "Bilinear", "enable")
    assert result_min[0].shape == (1, 32, 32, 3)

    # Test upscaling to larger size
    result_high = sut.resize_image(img, 512, 512, "Bilinear", "enable")
    assert result_high[0].shape == (1, 512, 512, 3)


def test_batch_processing(sut):
    """Test batch processing with multiple images."""
    img = torch.rand(2, 64, 64, 3)  # Batch size of 2
    result = sut.resize_image(img, 128, 128, "Magic Kernel", "enable")
    assert result[0].shape == (2, 128, 128, 3)


def test_nearest_interpolation_upscaling_fails(sut):
    """Test that Nearest interpolation raises an error when upsampling."""
    img = create_test_image()  # 64x64 image

    # Attempting to upscale with Nearest should fail
    with pytest.raises(ValueError) as exc_info:
        sut.resize_image(img, 128, 128, "Nearest", "enable")

    assert "Nearest interpolation does not support upsampling" in str(exc_info.value)


def test_output_dimensions(sut):
    """Test that out_width and out_height outputs match the actual image dimensions."""
    img = create_test_image()  # 64x64 image (aspect ratio = 1.0)
    target_width = 200
    target_height = 150

    result = sut.resize_image(img, target_width, target_height, "Bilinear", "enable")

    # Check return tuple has 3 elements
    assert len(result) == 3

    image_output, out_width, out_height = result

    # With aspect ratio preservation: 64x64 has ratio 1.0, target 200x150 has ratio 1.33
    # Target is "wider" so height is limiting → output is 150x150 (preserves 1:1 ratio)
    assert out_width == 150
    assert out_height == 150
    # Verify image dimensions match reported dimensions
    assert image_output.shape[1] == out_height  # height dimension
    assert image_output.shape[2] == out_width  # width dimension


def test_aspect_ratio_preservation_exact_match(sut):
    """Test that 768x512 → 1536x1024 produces exact output (same aspect ratio)."""
    # Source: 768×512 (aspect ratio = 1.5)
    # Target: 1536×1024 (aspect ratio = 1.5)
    # Since ratios match, output should be exactly 1536×1024
    img = torch.ones(1, 512, 768, 3)  # [batch, height, width, channels]
    result = sut.resize_image(img, 1536, 1024, "Bilinear", "enable")

    # Verify exact output dimensions
    assert result[0].shape == (1, 1024, 1536, 3), f"Expected (1, 1024, 1536, 3), got {result[0].shape}"
    assert result[1] == 1536, f"Expected out_width=1536, got {result[1]}"
    assert result[2] == 1024, f"Expected out_height=1024, got {result[2]}"


def test_aspect_ratio_preservation_downscale_width_limiting(sut):
    """Test that 768x512 → 256x256 produces 256x171 (width-limiting case)."""
    # Source: 768×512 (aspect ratio = 1.5)
    # Target: 256×256 (aspect ratio = 1.0)
    # Target is "taller" (1.0 < 1.5) → width is limiting
    # Output: width=256, height=round(256/1.5)=171
    img = torch.ones(1, 512, 768, 3)  # [batch, height, width, channels]
    result = sut.resize_image(img, 256, 256, "Bilinear", "enable")

    # Verify output dimensions
    assert result[0].shape == (1, 171, 256, 3), f"Expected (1, 171, 256, 3), got {result[0].shape}"
    assert result[1] == 256, f"Expected out_width=256, got {result[1]}"
    assert result[2] == 171, f"Expected out_height=171, got {result[2]}"
