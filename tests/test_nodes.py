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