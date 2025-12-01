import numpy as np
import os
from PIL import Image
from scipy.ndimage import map_coordinates
import nibabel as nib

def load_cbct_from_nii(nii_path):
    """
    Load CBCT volume from a NIfTI file.
    
    Parameters:
    -----------
    nii_path : str
        Path to .nii or .nii.gz file
        
    Returns:
    --------
    V : np.ndarray
        3D CBCT volume (D, H, W)
    voxel_spacing : tuple
        Voxel spacing (dz, dy, dx) in mm
    affine : np.ndarray
        Affine transformation matrix
    """
    # Load NIfTI file
    nii_img = nib.load(nii_path)
    V = nii_img.get_fdata()
    
    # Get voxel spacing from header
    voxel_spacing = nii_img.header.get_zooms()[:3]
    
    # Get affine matrix
    affine = nii_img.affine
    
    # Ensure correct shape (D, H, W)
    if len(V.shape) == 4:
        V = V[..., 0]  # Take first volume if 4D
    
    # Normalize to [0, 1]
    V = V.astype(np.float32)
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)
    
    return V, voxel_spacing, affine


def generate_axial_drr(V, voxel_spacing, H_out=512, W_out=512, beta=1e-4):
    """
    Generate DRR (Digitally Reconstructed Radiograph) from axial view.
    This simulates X-ray projection through the volume from top to bottom.
    
    Parameters:
    -----------
    V : np.ndarray
        3D CBCT volume (D, H, W) where D is depth (axial slices)
    voxel_spacing : tuple
        Voxel spacing (dz, dy, dx) in mm
    H_out : int
        Output image height
    W_out : int
        Output image width
    beta : float
        Attenuation coefficient
        
    Returns:
    --------
    drr : np.ndarray
        2D X-ray image (H_out, W_out)
    """
    D, H, W = V.shape
    dz, dy, dx = voxel_spacing
    
    print(f"Volume shape: {V.shape}")
    print(f"Voxel spacing: {voxel_spacing}")
    print(f"Generating {H_out}x{W_out} DRR from axial view...")
    
    # Create output image
    drr = np.zeros((H_out, W_out), dtype=np.float32)
    
    # For each pixel in output image
    for i in range(H_out):
        if i % 50 == 0:
            print(f"Processing row {i}/{H_out}")
            
        for j in range(W_out):
            # Map output pixel (i, j) to volume coordinates (y, x)
            y_vol = i * (H - 1) / (H_out - 1) if H_out > 1 else H / 2
            x_vol = j * (W - 1) / (W_out - 1) if W_out > 1 else W / 2
            
            # Cast ray through all depth slices (z direction)
            # Collect attenuation values along the ray
            attenuation_sum = 0.0
            
            for z_idx in range(D):
                # Get voxel value at (z_idx, y_vol, x_vol) using interpolation
                coords = np.array([[z_idx], [y_vol], [x_vol]])
                
                # Trilinear interpolation
                try:
                    value = map_coordinates(V, coords, order=1, mode='constant', cval=0)[0]
                except:
                    value = 0
                
                # Accumulate attenuation
                attenuation_sum += beta * value * dz
            
            # Apply Beer-Lambert law
            # T = exp(-attenuation_sum) is transmission
            # Pixel intensity = 1 - T (absorption)
            transmission = np.exp(-attenuation_sum)
            drr[i, j] = 1 - transmission
    
    return drr


def generate_axial_drr_fast(V, voxel_spacing, H_out=512, W_out=512, beta=1e-4):
    """
    Fast vectorized version of axial DRR generation.
    
    Parameters:
    -----------
    V : np.ndarray
        3D CBCT volume (D, H, W)
    voxel_spacing : tuple
        Voxel spacing (dz, dy, dx) in mm
    H_out : int
        Output image height
    W_out : int
        Output image width
    beta : float
        Attenuation coefficient
        
    Returns:
    --------
    drr : np.ndarray
        2D X-ray image (H_out, W_out)
    """
    D, H, W = V.shape
    dz, dy, dx = voxel_spacing
    
    print(f"Volume shape: {V.shape}")
    print(f"Voxel spacing: {voxel_spacing}")
    print(f"Generating {H_out}x{W_out} DRR from axial view (fast method)...")
    
    # Create coordinate grids for output image
    y_coords = np.linspace(0, H - 1, H_out)
    x_coords = np.linspace(0, W - 1, W_out)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Initialize accumulator
    attenuation_sum = np.zeros((H_out, W_out), dtype=np.float32)
    
    # Process each depth slice
    for z_idx in range(D):
        if z_idx % 20 == 0:
            print(f"Processing depth slice {z_idx}/{D}")
        
        # Create coordinates for this slice
        coords = np.array([
            np.full_like(yy, z_idx),
            yy,
            xx
        ]).reshape(3, -1)
        
        # Interpolate values
        values = map_coordinates(V, coords, order=1, mode='constant', cval=0)
        values = values.reshape(H_out, W_out)
        
        # Accumulate attenuation
        attenuation_sum += beta * values * dz
    
    # Apply Beer-Lambert law
    transmission = np.exp(-attenuation_sum)
    drr = 1 - transmission
    
    # Normalize to [0, 1]
    drr = (drr - drr.min()) / (drr.max() - drr.min() + 1e-8)
    
    return drr


def enhance_xray_contrast(drr, gamma=1.5, clip_percentile=1):
    """
    Enhance X-ray image contrast.
    
    Parameters:
    -----------
    drr : np.ndarray
        DRR image
    gamma : float
        Gamma correction value
    clip_percentile : float
        Percentile for contrast clipping
        
    Returns:
    --------
    enhanced : np.ndarray
        Enhanced image
    """
    # Clip extreme values
    lower = np.percentile(drr, clip_percentile)
    upper = np.percentile(drr, 100 - clip_percentile)
    
    drr_clipped = np.clip(drr, lower, upper)
    drr_normalized = (drr_clipped - lower) / (upper - lower + 1e-8)
    
    # Apply gamma correction
    enhanced = np.power(drr_normalized, gamma)
    
    return enhanced


def main(input_path, output_path='drr_xray_output.png', voxel_spacing=None, 
         H_out=512, W_out=512, beta=1e-4, gamma=1.5, use_fast=True):
    """
    Main function to generate X-ray DRR from CBCT axial view.
    
    Parameters:
    -----------
    input_path : str
        Path to .nii/.nii.gz file
    output_path : str
        Path to save output X-ray image
    voxel_spacing : tuple or None
        Voxel spacing (dz, dy, dx) in mm. If None, will be read from NIfTI header
    H_out : int
        Output image height
    W_out : int
        Output image width
    beta : float
        Attenuation coefficient (adjust for contrast)
    gamma : float
        Gamma correction for contrast enhancement
    use_fast : bool
        Use fast vectorized method
    """
    # Load CBCT from NIfTI file
    print(f"Loading CBCT from NIfTI file: {input_path}...")
    V, detected_spacing, affine = load_cbct_from_nii(input_path)
    
    # Use detected spacing if not provided
    if voxel_spacing is None:
        voxel_spacing = detected_spacing
        print(f"Using voxel spacing from NIfTI header: {voxel_spacing}")
    
    print(f"Loaded volume with shape: {V.shape}")
    print(f"Voxel spacing: {voxel_spacing}")
    
    # Generate DRR from axial view
    if use_fast:
        drr = generate_axial_drr_fast(V, voxel_spacing, H_out, W_out, beta)
    else:
        drr = generate_axial_drr(V, voxel_spacing, H_out, W_out, beta)
    
    # Enhance contrast
    print("Enhancing contrast...")
    drr_enhanced = enhance_xray_contrast(drr, gamma=gamma)
    
    # Invert for X-ray appearance (darker = more dense)
    drr_xray = 1 - drr_enhanced
    
    # Save output
    print(f"Saving output to {output_path}...")
    output_img = (drr_xray * 255).astype(np.uint8)
    Image.fromarray(output_img).save(output_path)
    
    # Also save non-inverted version
    output_path_normal = output_path.replace('.png', '_normal.png')
    output_img_normal = (drr_enhanced * 255).astype(np.uint8)
    Image.fromarray(output_img_normal).save(output_path_normal)
    
    print("Done!")
    print(f"Saved inverted X-ray to: {output_path}")
    print(f"Saved normal version to: {output_path_normal}")
    
    return drr_xray


if __name__ == "__main__":
    # Example usage with NIfTI file
    cbct_path = r"003_cbct.nii"
    output_path = r"drr_xray_axial.png"
    
    # Generate X-ray DRR from axial view
    # Adjust beta (higher = more contrast, typical range: 1e-5 to 1e-3)
    # Adjust gamma (higher = more contrast, typical range: 1.0 to 2.0)
    drr = main(
        cbct_path, 
        output_path=output_path, 
        H_out=512, 
        W_out=512, 
        beta=2e-4,  # Adjust for better contrast
        gamma=1.5,   # Adjust for better visualization
        use_fast=True
    )