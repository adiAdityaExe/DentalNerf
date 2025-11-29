import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime
from dNeRF import DentalNeRF, DentalXRayDataset, DentalNeRFTrainer
import nibabel as nib

def save_3d_visualization(volume, threshold=0.1, output_dir='.'):
    """Generate and save multiple views of the 3D volume"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter voxels above threshold
    voxel_array = volume > threshold
    x, y, z = np.where(voxel_array)
    
    if len(x) == 0:
        print("Warning: No voxels above threshold. Adjusting threshold...")
        threshold = np.percentile(volume, 50)  # Use median as threshold
        voxel_array = volume > threshold
        x, y, z = np.where(voxel_array)
    
    colors_values = volume[x, y, z]
    
    # Normalize colors for visualization
    colors_normalized = (colors_values - colors_values.min()) / (colors_values.max() - colors_values.min() + 1e-10)
    
    # Create figure with multiple subplots for different views
    fig = plt.figure(figsize=(20, 15))
    
    # View 1: 3D Scatter plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(x, y, z, c=colors_normalized, cmap='viridis', alpha=0.6, s=5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Dental Reconstruction - Isometric View')
    plt.colorbar(scatter, ax=ax1, label='Density')
    
    # View 2: Top view (XY plane)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(x, y, z, c=colors_normalized, cmap='viridis', alpha=0.6, s=5)
    ax2.view_init(elev=90, azim=0)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Top View (XY Plane)')
    
    # View 3: Front view (XZ plane)
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.scatter(x, y, z, c=colors_normalized, cmap='viridis', alpha=0.6, s=5)
    ax3.view_init(elev=0, azim=0)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Front View (XZ Plane)')
    
    # View 4: Side view (YZ plane)
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.scatter(x, y, z, c=colors_normalized, cmap='viridis', alpha=0.6, s=5)
    ax4.view_init(elev=0, azim=90)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('Side View (YZ Plane)')
    
    # View 5: 2D projection - XY
    ax5 = fig.add_subplot(2, 3, 5)
    xy_projection = np.max(volume, axis=2)
    im5 = ax5.imshow(xy_projection, cmap='viridis', aspect='auto')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_title('2D Projection - XY Plane')
    plt.colorbar(im5, ax=ax5, label='Max Density')
    
    # View 6: 2D projection - XZ
    ax6 = fig.add_subplot(2, 3, 6)
    xz_projection = np.max(volume, axis=1)
    im6 = ax6.imshow(xz_projection, cmap='viridis', aspect='auto')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Z')
    ax6.set_title('2D Projection - XZ Plane')
    plt.colorbar(im6, ax=ax6, label='Max Density')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'dental_3d_reconstruction_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D visualization saved to: {output_path}")
    
    plt.close()
    
    return output_path

def save_volume_slices(volume, output_dir='.', num_slices=10):
    """Save 2D slices through the volume"""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    depth = volume.shape[2]
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Dental Volume Slices (Along Z-axis)', fontsize=16)
    
    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx // 5, idx % 5]
        slice_data = volume[:, :, slice_idx]
        im = ax.imshow(slice_data, cmap='viridis', aspect='auto')
        ax.set_title(f'Slice {slice_idx}/{depth}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'dental_volume_slices_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Volume slices saved to: {output_path}")
    
    plt.close()
    
    return output_path

def save_as_nifti(volume, output_path):
    """
    Save volume as NIfTI file format (.nii)
    
    Args:
        volume: 3D numpy array
        output_path: Path to save the .nii file
    """
    # Create NIfTI image with identity affine matrix
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(volume, affine)
    
    # Save the NIfTI file
    nib.save(nifti_img, output_path)
    print(f"NIfTI file saved to: {output_path}")

def generate_and_save_3d_model(image_paths, output_dir='.', num_epochs=1000, grid_size=64):
    """
    Complete pipeline: Train NeRF and generate 3D model
    
    Args:
        image_paths: List of paths to dental X-ray images
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs
        grid_size: Resolution of 3D grid
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = DentalXRayDataset(image_paths)
    
    # Initialize model
    print("Initializing model...")
    model = DentalNeRF(hidden_dim=128, num_layers=6)
    
    # Initialize trainer
    trainer = DentalNeRFTrainer(model, dataset, device)
    
    # Train the model
    print(f"Training NeRF model for {num_epochs} epochs...")
    trainer.train(num_epochs=num_epochs)
    
    # Generate 3D volume
    print(f"Generating 3D volume with grid size {grid_size}...")
    volume = trainer.generate_3d_volume(grid_size=grid_size)
    
    # Normalize volume to 0-1 range for better visualization
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-10)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as NIfTI format
    nifti_path = os.path.join(output_dir, f'dental_3d_volume_{timestamp}.nii')
    save_as_nifti(volume, nifti_path)
    
    # Also save as compressed NIfTI
    nifti_gz_path = os.path.join(output_dir, f'dental_3d_volume_{timestamp}.nii.gz')
    save_as_nifti(volume, nifti_gz_path)
    
    # Save visualizations
    print("Generating visualizations...")
    viz_path = save_3d_visualization(volume, threshold=0.1, output_dir=output_dir)
    slices_path = save_volume_slices(volume, output_dir=output_dir, num_slices=10)
    
    # Save volume statistics
    stats_path = os.path.join(output_dir, f'volume_statistics_{timestamp}.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Dental 3D Volume Statistics\n")
        f.write(f"{'='*50}\n")
        f.write(f"Volume shape: {volume.shape}\n")
        f.write(f"Min density: {volume.min():.6f}\n")
        f.write(f"Max density: {volume.max():.6f}\n")
        f.write(f"Mean density: {volume.mean():.6f}\n")
        f.write(f"Std density: {volume.std():.6f}\n")
        f.write(f"Non-zero voxels: {np.count_nonzero(volume)} / {volume.size}\n")
        f.write(f"Non-zero percentage: {100 * np.count_nonzero(volume) / volume.size:.2f}%\n")
        f.write(f"\nOutput files:\n")
        f.write(f"  - NIfTI file: {nifti_path}\n")
        f.write(f"  - Compressed NIfTI: {nifti_gz_path}\n")
        f.write(f"  - 3D visualization: {viz_path}\n")
        f.write(f"  - Volume slices: {slices_path}\n")
    
    print(f"Statistics saved to: {stats_path}")
    
    print("\n" + "="*50)
    print("3D Model Generation Complete!")
    print("="*50)
    print(f"Output files:")
    print(f"  - NIfTI file: {nifti_path}")
    print(f"  - Compressed NIfTI: {nifti_gz_path}")
    print(f"  - 3D visualization: {viz_path}")
    print(f"  - Volume slices: {slices_path}")
    print(f"  - Statistics: {stats_path}")
    
    return volume, nifti_path, nifti_gz_path, viz_path, slices_path

def main():
    # Configuration
    image_paths = [
        "3.png",
        "4.png",
    ]
    
    output_dir = "output_3d_models"
    
    # Check if images exist
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Warning: Image not found: {path}")
    
    # Generate 3D model
    volume, nifti_path, nifti_gz_path, viz_path, slices_path = generate_and_save_3d_model(
        image_paths=image_paths,
        output_dir=output_dir,
        num_epochs=1000,
        grid_size=64
    )
    
    print("\nTo load the NIfTI volume later, use:")
    print(f"  import nibabel as nib")
    print(f"  nifti_img = nib.load('{nifti_path}')")
    print(f"  volume = nifti_img.get_fdata()")

if __name__ == "__main__":
    main()