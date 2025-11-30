import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def load_cbct_nifti(nifti_path):
    """
    Load CBCT image from NIfTI file
    
    Args:
        nifti_path: Path to the .nii or .nii.gz file
    
    Returns:
        volume: 3D numpy array
        affine: Affine transformation matrix
        header: NIfTI header
    """
    print(f"Loading CBCT image from: {nifti_path}")
    nifti_img = nib.load(nifti_path)
    volume = nifti_img.get_fdata()
    affine = nifti_img.affine
    header = nifti_img.header
    
    print(f"Volume shape: {volume.shape}")
    print(f"Data type: {volume.dtype}")
    print(f"Value range: [{volume.min():.2f}, {volume.max():.2f}]")
    
    return volume, affine, header

def slice_and_save_cbct(volume, output_dir, axis='axial', image_format='png'):
    """
    Slice CBCT volume along specified axis and save individual slices
    
    Args:
        volume: 3D numpy array
        output_dir: Directory to save slices
        axis: Slicing axis - 'axial' (z), 'sagittal' (x), or 'coronal' (y)
        image_format: Format to save images ('png', 'jpg', etc.)
    
    Returns:
        slice_paths: List of paths to saved slices
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine slicing axis
    axis_map = {
        'axial': 2,      # Along Z-axis (top to bottom)
        'sagittal': 0,   # Along X-axis (left to right)
        'coronal': 1     # Along Y-axis (front to back)
    }
    
    slice_axis = axis_map.get(axis.lower(), 2)
    num_slices = volume.shape[slice_axis]
    
    print(f"\nSlicing along {axis} axis...")
    print(f"Number of slices: {num_slices}")
    
    slice_paths = []
    
    # Normalize volume for visualization (0-255)
    volume_normalized = ((volume - volume.min()) / (volume.max() - volume.min() + 1e-10) * 255).astype(np.uint8)
    
    for i in range(num_slices):
        # Extract slice based on axis
        if slice_axis == 0:  # Sagittal
            slice_data = volume_normalized[i, :, :]
        elif slice_axis == 1:  # Coronal
            slice_data = volume_normalized[:, i, :]
        else:  # Axial
            slice_data = volume_normalized[:, :, i]
        
        # Save slice
        slice_filename = f"{axis}_slice_{i:04d}.{image_format}"
        slice_path = os.path.join(output_dir, slice_filename)
        
        plt.imsave(slice_path, slice_data, cmap='gray')
        slice_paths.append(slice_path)
    
    print(f"Saved {num_slices} slices to: {output_dir}")
    
    return slice_paths

def reconstruct_nifti_from_slices(slice_dir, output_nifti_path, axis='axial', 
                                   original_affine=None, original_header=None,
                                   image_format='png'):
    """
    Reconstruct NIfTI volume from saved slices
    
    Args:
        slice_dir: Directory containing slice images
        output_nifti_path: Path to save reconstructed .nii file
        axis: Original slicing axis
        original_affine: Original affine matrix (optional)
        original_header: Original NIfTI header (optional)
        image_format: Format of slice images
    
    Returns:
        reconstructed_volume: 3D numpy array
    """
    print(f"\nReconstructing NIfTI from slices in: {slice_dir}")
    
    # Get sorted list of slice files
    slice_files = sorted([f for f in os.listdir(slice_dir) 
                         if f.endswith(f'.{image_format}') and f.startswith(f'{axis}_slice_')])
    
    if len(slice_files) == 0:
        raise ValueError(f"No slice files found in {slice_dir}")
    
    print(f"Found {len(slice_files)} slices")
    
    # Read first slice to get dimensions
    first_slice = plt.imread(os.path.join(slice_dir, slice_files[0]))
    if len(first_slice.shape) == 3:
        first_slice = first_slice[:, :, 0]  # Convert RGB to grayscale if needed
    
    # Initialize volume array
    axis_map = {
        'axial': 2,
        'sagittal': 0,
        'coronal': 1
    }
    slice_axis = axis_map.get(axis.lower(), 2)
    
    # Determine volume shape
    if slice_axis == 0:  # Sagittal
        volume_shape = (len(slice_files), first_slice.shape[0], first_slice.shape[1])
    elif slice_axis == 1:  # Coronal
        volume_shape = (first_slice.shape[0], len(slice_files), first_slice.shape[1])
    else:  # Axial
        volume_shape = (first_slice.shape[0], first_slice.shape[1], len(slice_files))
    
    reconstructed_volume = np.zeros(volume_shape, dtype=np.float32)
    
    # Load all slices
    print("Loading slices...")
    for i, slice_file in enumerate(slice_files):
        slice_path = os.path.join(slice_dir, slice_file)
        slice_data = plt.imread(slice_path)
        
        # Convert to grayscale if needed
        if len(slice_data.shape) == 3:
            slice_data = slice_data[:, :, 0]
        
        # Place slice in volume
        if slice_axis == 0:  # Sagittal
            reconstructed_volume[i, :, :] = slice_data
        elif slice_axis == 1:  # Coronal
            reconstructed_volume[:, i, :] = slice_data
        else:  # Axial
            reconstructed_volume[:, :, i] = slice_data
    
    print(f"Reconstructed volume shape: {reconstructed_volume.shape}")
    
    # Create affine matrix if not provided
    if original_affine is None:
        original_affine = np.eye(4)
    
    # Create NIfTI image
    if original_header is not None:
        nifti_img = nib.Nifti1Image(reconstructed_volume, original_affine, header=original_header)
    else:
        nifti_img = nib.Nifti1Image(reconstructed_volume, original_affine)
    
    # Save reconstructed NIfTI
    nib.save(nifti_img, output_nifti_path)
    print(f"Reconstructed NIfTI saved to: {output_nifti_path}")
    
    return reconstructed_volume

def visualize_comparison(original_volume, reconstructed_volume, num_samples=5):
    """
    Visualize comparison between original and reconstructed volumes
    """
    depth = original_volume.shape[2]
    slice_indices = np.linspace(0, depth-1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
    fig.suptitle('Original vs Reconstructed CBCT Slices', fontsize=16)
    
    for idx, slice_idx in enumerate(slice_indices):
        # Original
        axes[0, idx].imshow(original_volume[:, :, slice_idx], cmap='gray')
        axes[0, idx].set_title(f'Original - Slice {slice_idx}')
        axes[0, idx].axis('off')
        
        # Reconstructed
        axes[1, idx].imshow(reconstructed_volume[:, :, slice_idx], cmap='gray')
        axes[1, idx].set_title(f'Reconstructed - Slice {slice_idx}')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main pipeline for CBCT slicing and reconstruction
    """
    # Configuration
    input_nifti_path = "003_cbct.nii"  # Replace with your CBCT file path
    base_output_dir = "cbct_slicing_output"
    
    # Create timestamp for unique folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"slices_{timestamp}")
    
    # Choose slicing axis: 'axial', 'sagittal', or 'coronal'
    slicing_axis = 'axial'
    
    print("="*60)
    print("CBCT Slicing and Reconstruction Pipeline")
    print("="*60)
    
    # Step 1: Load CBCT NIfTI file
    try:
        volume, affine, header = load_cbct_nifti(input_nifti_path)
    except FileNotFoundError:
        print(f"\nError: File not found: {input_nifti_path}")
        print("Please update 'input_nifti_path' with your CBCT file path.")
        return
    
    # Step 2: Slice and save
    slice_paths = slice_and_save_cbct(
        volume=volume,
        output_dir=output_dir,
        axis=slicing_axis,
        image_format='png'
    )
    
    # Step 3: Reconstruct from slices
    reconstructed_nifti_path = os.path.join(
        base_output_dir, 
        f"reconstructed_cbct_{timestamp}.nii"
    )
    
    reconstructed_volume = reconstruct_nifti_from_slices(
        slice_dir=output_dir,
        output_nifti_path=reconstructed_nifti_path,
        axis=slicing_axis,
        original_affine=affine,
        original_header=header,
        image_format='png'
    )
    
    # Step 4: Verify reconstruction
    print("\n" + "="*60)
    print("Verification")
    print("="*60)
    print(f"Original volume shape: {volume.shape}")
    print(f"Reconstructed volume shape: {reconstructed_volume.shape}")
    print(f"Shapes match: {volume.shape == reconstructed_volume.shape}")
    
    # Calculate reconstruction error
    if volume.shape == reconstructed_volume.shape:
        # Normalize both to 0-1 range for fair comparison
        vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-10)
        recon_norm = (reconstructed_volume - reconstructed_volume.min()) / (reconstructed_volume.max() - reconstructed_volume.min() + 1e-10)
        
        mse = np.mean((vol_norm - recon_norm) ** 2)
        print(f"Mean Squared Error: {mse:.6f}")
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"Slices saved to: {output_dir}")
    print(f"Reconstructed NIfTI: {reconstructed_nifti_path}")
    print(f"Total slices: {len(slice_paths)}")
    
    # Optional: Visualize comparison
    # Uncomment the line below to see visual comparison
    # visualize_comparison(volume, reconstructed_volume, num_samples=5)

if __name__ == "__main__":
    main()