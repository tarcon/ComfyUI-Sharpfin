# ComfyUI Sharpfin

ComfyUI node to change image dimensions with Torchvision transforms built for accuracy, visual quality, and speed.

It can resize an image with rare resampling algorithms that achieve higher quality than the standards.

This node uses code from the [Sharpfin library](https://github.com/drhead/Sharpfin) licensed under Apache 2.0.

## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started).
2. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
3. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under
   `ComfyUI/custom_nodes`.
4. Restart ComfyUI.

## Features

- High-quality image resampling with performant Pytorch libs
- Optional sRGB conversion: For precise colour calculation and correct colour space conversion
- Ten different interpolation kernels:
    - Nearest
    - Bilinear
    - Mitchell
    - Catmull-Rom
    - B-Spline
    - Lanczos2
    - Lanczos3
    - Magic Kernel
    - Magic Kernel Sharp 2013
    - Magic Kernel Sharp 2021

**More Details**: Confer to the documentation of the underlying library [Sharpfin](https://github.com/drhead/Sharpfin)

## Changelog

### Version 2.0.0

- Keeps aspect ratios as calculated from largest dimension (Change from 1.0.0 behaviour)
- Allows unstepped target dimensions input
- Outputs resulting width and height for further processing in ComfyUI
- Node info documentation

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd sharpfin
pip install .
pre-commit install
```

## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. It is recommended to unit test your custom
node.

- [build-pipeline.yml](.github/workflows/build.yml) will run pytest and linter on any open PRs
