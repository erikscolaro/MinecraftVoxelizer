# MinecraftModelVoxelizer

A tool that converts 3D STL models into voxelized layer images, perfect for building structures in Minecraft layer by layer.

## Overview

This project takes 3D models in STL format and converts them into a series of 2D images, with each image representing a horizontal slice (layer) of the model. This makes it easy to visualize how to build complex structures in Minecraft or similar voxel-based games by following the generated layer plans.

## Features

- Converts STL files to voxel representations
- Creates layer-by-layer image guides with grid lines
- Removes internal voxels to create shell models (saving resources when building)
- Automatic mesh repair for non-watertight models
- Configurable voxel resolution

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/MinecraftVoxelizer.git
   cd MinecraftVoxelizer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

   If the requirements.txt file doesn't exist, you'll need to install these packages:
   ```
   pip install numpy matplotlib trimesh
   ```

## Usage

1. On first run, the program will create an `input` folder in the project directory.
2. Place your STL files in the `input` folder.
3. Run the main script:
   ```
   python main.py
   ```
4. Follow the prompts to select your STL file.
5. The program will process the model and save layer images in a subfolder of the `output` directory.
6. Use the generated images as a building guide in Minecraft, starting from the bottom layer.

## Configuration

You can modify the following parameters in `main.py` to customize the output:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pitch` | Voxel size (smaller = higher resolution) | 1 |
| `voxelization_method` | Method used (1=local, 2=full) | 2 |
| `image_scale_factor` | Size of output images | 0.25 |
| `min_image_size` | Minimum image size in inches | 8 |
| `image_dpi` | DPI for saved images | 100 |
| `minor_grid_step` | Step size for minor grid lines | 1 |
| `major_grid_step` | Step size for major grid lines | 10 |

### Advanced Parameters

The program uses Taubin smoothing for mesh preparation. It's recommended not to modify these parameters unless you understand their impact:

- `taubin_lambda` (0.5)
- `taubin_nu` (-0.53)
- `taubin_iterations` (10)

## Tips for Best Results

- For optimal results, use watertight STL models (fully enclosed surfaces without holes).
- Adjust the pitch value based on your model size - smaller values produce more detailed voxelizations but result in more layers.
- If your model has very fine details, you may need to scale it up before voxelization.
- White pixels in the output images represent empty space, black pixels show where blocks should be placed.

## Troubleshooting

- If you encounter "out of memory" errors, try increasing the pitch value for a lower-resolution voxelization.
- Models with very thin features may not voxelize properly at larger pitch values.
- The automatic mesh repair can help with some issues, but severely broken meshes may require repair in a 3D modeling program first.

## License

[MIT License](LICENSE)
