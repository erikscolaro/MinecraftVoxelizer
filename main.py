# Standard library imports
import os
import time
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from trimesh.smoothing import filter_taubin
from trimesh.repair import fill_holes, fix_inversion
from trimesh.voxel import creation

# Global variables for configuration
voxelization_method = 2
debug = False

# Voxelization parameters
pitch = 1  # Adjust this value based on your needs (smaller value = higher resolution)
voxel_radius_factor = 6  # Divisor for calculating voxel radius in local voxelization

# Mesh repair and smoothing parameters
taubin_lambda = 0.5  # Lambda parameter for Taubin smoothing
taubin_nu = -0.53  # Nu parameter for Taubin smoothing
taubin_iterations = 10  # Iterations for Taubin smoothing

# Image output parameters
minor_grid_step = 1  # Step size for minor grid lines
major_grid_step = 10  # Step size for major grid lines
image_scale_factor = 0.25  # Scale factor for image size
min_image_size = 8  # Minimum image size in inches
image_dpi = 100  # DPI for saved images

# Example of working with paths
input_dir = Path("./input")
output_dir = Path("./output")

def setup(input_dir, output_dir):
  """Create input and output directories if they don't exist."""
  # Create input directory if it doesn't exist
  if not input_dir.exists():
    input_dir.mkdir(parents=True)
  
  # Create output directory if it doesn't exist
  if not output_dir.exists():
    output_dir.mkdir(parents=True)

def select_file_from_directory(directory, file_extension="*.stl"):
  """
  Prompt user to select a file from a directory with the specified file extension.
  
  Args:
      directory: Path object representing the directory to search
      file_extension: File extension pattern to match (default: "*.stl")
      
  Returns:
      tuple: (selected file Path object, absolute path to the selected file)
  """
  input_files = list(directory.glob(file_extension))

  if not input_files:
    print(f"No {file_extension[1:]} files found in the directory. Please add some files and try again.")
    exit()

  print(f"Available {file_extension[1:]} files:")
  for i, file in enumerate(input_files):
    print(f"{i+1}. {file.name}")

  while True:
    try:
      selection = int(input("Select a file by number: ")) - 1
      if 0 <= selection < len(input_files):
        selected_file = input_files[selection]
        selected_file_path = selected_file.resolve()
        print(f"Selected: {selected_file.name}")
        return selected_file, selected_file_path
      else:
        print(f"Please enter a number between 1 and {len(input_files)}")
    except ValueError:
      print("Please enter a valid number")

def prepare_mesh_for_voxelization(mesh):
  """
  Prepare a mesh for voxelization by ensuring it's watertight and applying repairs.
  
  Args:
    mesh: A trimesh mesh object
  
  Returns:
    The prepared mesh object
  """
  # Ensure the mesh is watertight for better voxelization
  if not mesh.is_watertight:
    print("Warning: Mesh is not watertight, which may affect voxelization quality")
    print("Attempting to repair the model...")
    
    print("Removing unreferenced vertices...")
    mesh.remove_unreferenced_vertices()
    
    print("Removing degenerate faces...")
    mesh.update_faces(mesh.nondegenerate_faces())
    
    print("Removing duplicate faces...")
    mesh.update_faces(mesh.unique_faces())
    
    print("Attempting to fill small holes...")
    mesh.fill_holes()
    
    print("Fixing mesh face inversions...")
    trimesh.repair.fix_inversion(mesh)
    
    print("Performing additional hole filling...")
    trimesh.repair.fill_holes(mesh)
    
    # Check if the mesh is now watertight after repairs
    if mesh.is_watertight:
      print("Mesh repair was successful! The mesh is now watertight.")
      # Apply Laplacian smoothing to the mesh for better quality
      print("Applying Laplacian smoothing to improve mesh quality...")
      
      # Create a copy of the original mesh for comparison
      original_vertex_count = len(mesh.vertices)
      original_face_count = len(mesh.faces)
      
      # Apply smoothing with specified parameters
      filter_taubin(mesh, lamb=taubin_lambda, nu=taubin_nu, iterations=taubin_iterations)
      
      print(f"Laplacian smoothing complete: {original_vertex_count} vertices processed")
      print(f"Mesh now has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    else:
      print("Warning: Mesh repair was only partially successful. The mesh still has issues.")
  
  # Additional mesh quality info
  print(f"Volume: {mesh.volume:.2f} cubic units")
  print(f"Surface area: {mesh.area:.2f} square units")
  
  return mesh

def voxelize_mesh(mesh, method=2, pitch=1):
  """
  Voxelize a mesh using the specified method and pitch.
  
  Args:
    mesh: A trimesh mesh object
    method: Voxelization method (1=local, 2=full)
    pitch: Voxel size (smaller value = higher resolution)
    
  Returns:
    A voxel grid object
  """
  # Get mesh bounds to understand its size
  bounds = mesh.bounds
  dimensions = bounds[1] - bounds[0]
  
  # Use match statement for voxelization method selection
  match method:
    case 1:
      # Voxelize the mesh with the specified pitch
      print(f"Voxelizing with pitch={pitch}...")
      # Define the central point of interest and radius for local voxelization
      # Default to the center of the mesh if not specified
      center_point = mesh.centroid
      voxel_radius = int(sum(dimensions) / voxel_radius_factor)  # Default radius covers the whole mesh, converted to int
      print(f"Local voxelization centered at {center_point} with radius {voxel_radius}")
      return creation.local_voxelize(mesh, point=center_point, radius=voxel_radius, pitch=pitch)
    case 2:
      print(f"Mesh dimensions: {dimensions} units")
      estimated_voxels = (dimensions / pitch).astype(int)
      print(f"Estimated voxel grid size: {estimated_voxels}")
      # Voxelize the mesh with the specified pitch
      print(f"Voxelizing with pitch={pitch}...")
      return mesh.voxelized(pitch=pitch)

def remove_internal_voxels(matrix):
  """
  Create a filtered matrix where internal voxels are removed, keeping only the shell.
  
  Args:
    matrix: 3D numpy array of voxels
    
  Returns:
    Filtered matrix with internal voxels removed
  """
  filtered_matrix = matrix.copy()
  
  # Skip the outer layer of voxels to avoid index errors
  for z in range(1, matrix.shape[2] - 1):
    for y in range(1, matrix.shape[0] - 1):
      for x in range(1, matrix.shape[1] - 1):
        # Check if current voxel is filled
        if matrix[y, x, z]:
          # Check all 6 adjacent neighbors (up, down, left, right, front, back)
          if (matrix[y+1, x, z] and matrix[y-1, x, z] and
            matrix[y, x+1, z] and matrix[y, x-1, z] and
            matrix[y, x, z+1] and matrix[y, x, z-1]):
            # Voxel is completely surrounded, set it to 0 in the filtered matrix
            filtered_matrix[y, x, z] = False
  
  return filtered_matrix

def save_voxel_layers_as_images(matrix, output_dir, base_filename):
  """
  Save each layer of a voxel matrix as a grid-lined image.
  
  Args:
    matrix: 3D numpy array of voxels
    output_dir: Path object representing the output directory
    base_filename: Base name for the output files
  
  Returns:
    Path to the output directory
  """
  # Get dimensions
  voxel_shape = matrix.shape
  print(f"Voxel dimensions: {voxel_shape}")

  # Create a subfolder in output directory named after the input file
  output_subfolder = output_dir / base_filename
  if output_subfolder.exists():
    # Clear the directory by removing all files
    for file in output_subfolder.glob("*"):
      file.unlink()  # Delete the file
  else:
    output_subfolder.mkdir(parents=True)

  # Pre-calculate ticks and labels once instead of for each layer
  x_major_ticks = [i - 0.5 for i in range(0, matrix.shape[1] + 1, major_grid_step)]
  x_minor_ticks = [i - 0.5 for i in range(0, matrix.shape[1] + 1, minor_grid_step)]
  y_major_ticks = [i - 0.5 for i in range(0, matrix.shape[0] + 1, major_grid_step)]
  y_minor_ticks = [i - 0.5 for i in range(0, matrix.shape[0] + 1, minor_grid_step)]

  x_major_labels = [str(int(x + 0.5)) if (x + 0.5) % major_grid_step == 0 else '' for x in x_major_ticks]
  y_major_labels = [str(int(y + 0.5)) if (y + 0.5) % major_grid_step == 0 else '' for y in y_major_ticks]

  # Calculate figure size dynamically based on matrix dimensions
  fig_width = max(min_image_size, matrix.shape[1] * image_scale_factor)
  fig_height = max(min_image_size, matrix.shape[0] * image_scale_factor)

  # Save each layer as an image
  for i in range(voxel_shape[2]):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=image_dpi)
    ax.imshow(~matrix[:, :, i], cmap="gray")
    ax.set_title(f"Layer {i+1} of {voxel_shape[2]}")
    # Set up grid and ticks
    ax.set_xticks(x_major_ticks)
    ax.set_xticklabels(x_major_labels)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_yticks(y_major_ticks)
    ax.set_yticklabels(y_major_labels)
    ax.set_yticks(y_minor_ticks, minor=True)
    # Configure grid
    ax.grid(which='major', color='gray', linewidth=0.8)
    ax.grid(which='minor', color='lightgray', linewidth=0.3)
    # Set axis limits
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    output_path = output_subfolder / f"layer_{i+1:04d}.png"
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
  
  print(f"Saved {voxel_shape[2]} layer images to {output_subfolder}")
  return output_subfolder

def process_mesh(selected_file, selected_file_path):
  # Load the mesh and measure time
  start_time = time.time()
  print(f"Loading mesh: {selected_file_path}")
  mesh = trimesh.load(selected_file_path)
  
  # Prepare the mesh for voxelization
  mesh = prepare_mesh_for_voxelization(mesh)
  
  # Visualize the original mesh
  if debug: mesh.show()
  
  # Voxelize the mesh
  voxel_grid = voxelize_mesh(mesh, method=voxelization_method, pitch=pitch)
  matrix = voxel_grid.matrix
  print(f"Voxelization completed in {time.time() - start_time:.2f} seconds")
  print(f"Voxel matrix shape: {matrix.shape}")
  
  # Create shell model and save results
  process_voxel_matrix(matrix, selected_file)

def process_voxel_matrix(matrix, selected_file):
  # Remove internal voxels to create a shell model
  print("Removing internal voxels...")
  original_count = matrix.sum()
  filtered_matrix = remove_internal_voxels(matrix)
  filtered_count = filtered_matrix.sum()
  
  # Log statistics
  print(f"Original filled voxels: {original_count}")
  print(f"Filtered filled voxels: {filtered_count}")
  print(f"Removed internal voxels: {original_count - filtered_count}")
  
  # Save voxel layers as images
  output_path = save_voxel_layers_as_images(filtered_matrix, output_dir, selected_file.stem)
# Execute the main function

def main():
  # Setup the environment
  setup(input_dir, output_dir)
  
  # File selection
  selected_file, selected_file_path = select_file_from_directory(input_dir)
  
  # Process the mesh
  process_mesh(selected_file, selected_file_path)

if __name__ == "__main__":
  main()
