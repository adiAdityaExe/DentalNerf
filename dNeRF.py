import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
from skimage import io, transform
import os
from tqdm import tqdm

class DentalNeRF(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=8):
        super(DentalNeRF, self).__init__()
        
        # Positional encoding for 3D coordinates
        self.pos_enc_dim = 10
        self.input_dim = 3 * 2 * self.pos_enc_dim + 3
        
        # Main MLP network
        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.backbone = nn.Sequential(*layers)
        
        # Output layers
        self.density_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Linear(hidden_dim, 3)
        
    def positional_encoding(self, x):
        """Apply positional encoding to input coordinates"""
        encodings = [x]  # Start with original coordinates
        for i in range(self.pos_enc_dim):
            for fn in [torch.sin, torch.cos]:
                encodings.append(fn(2.0 ** i * x))
        return torch.cat(encodings, dim=-1)
    
    def forward(self, x):
        # x: [batch_size, 3] - 3D coordinates
        encoded = self.positional_encoding(x)
        features = self.backbone(encoded)
        
        density = F.relu(self.density_head(features))
        color = torch.sigmoid(self.color_head(features))
        
        return density, color

class DentalXRayDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256)):
        self.image_paths = image_paths
        self.image_size = image_size
        self.images = self.load_images()
        
    def load_images(self):
        images = []
        for path in self.image_paths:
            img = io.imread(path)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = transform.resize(img, self.image_size)
            images.append(img)
        return np.array(images)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.images[idx])

class DentalNeRFTrainer:
    def __init__(self, model, dataset, device='cuda'):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
    def generate_rays(self, img_idx, num_rays=1024):
        """Generate rays for volume rendering"""
        H, W = self.dataset.image_size
        img = self.dataset[img_idx]
        
        # Random sample pixels
        i = torch.randint(0, H, (num_rays,))
        j = torch.randint(0, W, (num_rays,))
        
        # Get pixel values as target
        target_pixels = img[i, j].unsqueeze(-1).repeat(1, 3)  # Convert to RGB
        
        # Generate ray directions (simplified)
        ray_dirs = torch.stack([
            (j - W/2) / (W/2),  # x coordinate
            (i - H/2) / (H/2),  # y coordinate
            torch.ones(num_rays)  # z coordinate
        ], dim=-1).to(self.device)
        
        # Generate sample points along rays
        num_samples = 64
        t_vals = torch.linspace(0, 1, num_samples).to(self.device)
        
        # Sample points in 3D space
        sample_points = ray_dirs.unsqueeze(1) * t_vals.unsqueeze(0).unsqueeze(-1)
        sample_points = sample_points.reshape(-1, 3)
        
        return sample_points, target_pixels.to(self.device)
    
    def volume_render(self, densities, colors, num_rays, num_samples):
        """Simple volume rendering"""
        densities = densities.reshape(num_rays, num_samples, 1)
        colors = colors.reshape(num_rays, num_samples, 3)
        
        # Alpha composition
        alphas = 1 - torch.exp(-F.relu(densities) * 0.01)
        transmittance = torch.cumprod(1 - alphas + 1e-10, dim=1)
        weights = alphas * transmittance
        
        rendered_pixels = torch.sum(weights * colors, dim=1)
        
        return rendered_pixels
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for img_idx in range(len(self.dataset)):
            # Generate rays and sample points
            sample_points, target_pixels = self.generate_rays(img_idx)
            num_rays = 1024
            num_samples = 64
            
            # Forward pass
            densities, colors = self.model(sample_points)
            
            # Volume rendering
            rendered_pixels = self.volume_render(densities, colors, num_rays, num_samples)
            
            # Compute loss
            loss = F.mse_loss(rendered_pixels, target_pixels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.dataset)
    
    def train(self, num_epochs=1000):
        for epoch in tqdm(range(num_epochs)):
            loss = self.train_epoch()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def generate_3d_volume(self, grid_size=64):
        """Generate a 3D volume from the trained model"""
        self.model.eval()
        
        # Create 3D grid
        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(-1, 1, grid_size)
        z = torch.linspace(-1, 1, grid_size)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        
        # Predict density for each point
        volume = []
        batch_size = 4096
        
        with torch.no_grad():
            for i in range(0, len(grid_points), batch_size):
                batch_points = grid_points[i:i+batch_size].to(self.device)
                densities, _ = self.model(batch_points)
                volume.append(densities.cpu())
        
        volume = torch.cat(volume).reshape(grid_size, grid_size, grid_size)
        return volume.numpy()

def visualize_3d_volume(volume, threshold=0.1):
    """Visualize the 3D volume using matplotlib"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create voxel grid
    voxel_array = volume > threshold
    
    # Plot voxels
    x, y, z = np.where(voxel_array)
    ax.scatter(x, y, z, c=volume[x, y, z], cmap='viridis', alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Dental Reconstruction')
    
    plt.show()

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load your dental X-ray images
    # Replace with your actual image paths
    image_paths = [
        "3.png",
        "4.png",
        # Add more images from different angles if available
    ]
    
    # Create dataset
    dataset = DentalXRayDataset(image_paths)
    
    # Initialize model
    model = DentalNeRF(hidden_dim=128, num_layers=6)
    
    # Initialize trainer
    trainer = DentalNeRFTrainer(model, dataset, device)
    
    # Train the model
    print("Training NeRF model...")
    trainer.train(num_epochs=1000)
    
    # Generate 3D volume
    print("Generating 3D volume...")
    volume = trainer.generate_3d_volume(grid_size=64)
    
    # Visualize results
    visualize_3d_volume(volume)
    
    # Save the 3D volume
    np.save("dental_3d_volume.npy", volume)
    print("3D volume saved as 'dental_3d_volume.npy'")

if __name__ == "__main__":
    main()