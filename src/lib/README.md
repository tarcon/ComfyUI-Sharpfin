## ⚠️ Modified Version for ComfyUI-Sharpfin

**This uses code from the Sharpfin library.**

### Original Author & License
- **Original Library**: [Sharpfin](https://github.com/drhead/Sharpfin) by [@Sharpfin](https://github.com/drhead)
- **License**: Apache License 2.0 (see LICENSE file)

### Modifications Made (January 2025)
The following modifications were made to enable cross-platform compatibility, specifically to support **macOS** where Triton is not available:

1. **Removed Triton dependency**: The library originally used Triton for GPU-accelerated sparse matrix operations. These files have been removed:
   - `triton_functional.py` - Triton JIT-compiled kernel functions
   - `sparse_backend.py` - Sparse matrix operations using Triton

2. **Updated `functional.py`**: Removed import of `downscale_sparse` from `triton_functional`. The `use_sparse` parameter now issues a warning and falls back to the dense (pure PyTorch) implementation.

3. **Updated `transforms.py`**: Removed import of `downscale_sparse` from `triton_functional`. The `downscale_sparse` method now falls back to the regular `downscale` method with a warning.

### Impact of Changes
- **Removed**: The Triton-based sparse GPU optimization (which provided ~7x speedup over naive GPU implementation)
- **Preserved**: All core functionality including:
  - All resize kernels (Nearest, Bilinear, Mitchell, Catmull-Rom, B-Spline, Lanczos2, Lanczos3, Magic Kernel, Magic Kernel Sharp 2013/2021)
  - sRGB to linear RGB color space conversion
  - Quantization and dithering options
  - All transforms (Scale, ApplyCMS, AlphaComposite, AspectRatioCrop)

The pure PyTorch implementation works on all platforms including macOS, Windows, and Linux.

---
