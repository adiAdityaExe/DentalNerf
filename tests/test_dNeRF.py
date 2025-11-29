import pytest
import torch
import sys
import os

# Add parent directory to path to import dNeRF
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dNeRF import DentalNeRF, DentalXRayDataset, DentalNeRFTrainer


class TestDentalNeRF:
    """Test suite for DentalNeRF model"""
    
    def test_model_initialization(self):
        """Test model can be initialized with correct dimensions"""
        model = DentalNeRF(hidden_dim=128, num_layers=6)
        assert model.pos_enc_dim == 10
        assert model.input_dim == 3 * 2 * 10 + 3  # Should be 63
        
    def test_positional_encoding_output_shape(self):
        """Test positional encoding produces correct output shape"""
        model = DentalNeRF(hidden_dim=128, num_layers=6)
        x = torch.randn(100, 3)  # Batch of 100 3D points
        encoded = model.positional_encoding(x)
        
        # Should output original 3 dims + encoded dims (3 * 2 * pos_enc_dim)
        expected_dim = 3 + 3 * 2 * model.pos_enc_dim
        assert encoded.shape == (100, expected_dim), f"Expected shape (100, {expected_dim}), got {encoded.shape}"
        
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shapes"""
        model = DentalNeRF(hidden_dim=128, num_layers=6)
        x = torch.randn(100, 3)  # Batch of 100 3D points
        
        density, color = model(x)
        
        assert density.shape == (100, 1), f"Expected density shape (100, 1), got {density.shape}"
        assert color.shape == (100, 3), f"Expected color shape (100, 3), got {color.shape}"
        
    def test_forward_pass_output_range(self):
        """Test forward pass outputs are in valid ranges"""
        model = DentalNeRF(hidden_dim=128, num_layers=6)
        x = torch.randn(100, 3)
        
        density, color = model(x)
        
        # Density should be non-negative (ReLU applied)
        assert torch.all(density >= 0), "Density values should be non-negative"
        
        # Color should be in [0, 1] (sigmoid applied)
        assert torch.all(color >= 0) and torch.all(color <= 1), "Color values should be in [0, 1]"
        
    def test_positional_encoding_includes_original_coords(self):
        """Test that positional encoding concatenates original coordinates"""
        model = DentalNeRF(hidden_dim=128, num_layers=6)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        
        encoded = model.positional_encoding(x)
        
        # The encoded output should include the original coordinates
        # Check dimension: should be 3 (original) + 3*2*10 (encoded) = 63
        assert encoded.shape[1] == 63, f"Expected 63 dimensions, got {encoded.shape[1]}"


class TestDentalXRayDataset:
    """Test suite for DentalXRayDataset"""
    
    def test_dataset_initialization_empty(self):
        """Test dataset can be initialized with empty image list"""
        dataset = DentalXRayDataset([], image_size=(128, 128))
        assert len(dataset) == 0


class TestDimensionBug:
    """Test to reproduce the dimension mismatch bug"""
    
    def test_dimension_mismatch_bug(self):
        """This test should fail with the current implementation, revealing the bug"""
        model = DentalNeRF(hidden_dim=128, num_layers=6)
        
        # Simulate the sample_points from generate_rays (65536 = 1024 rays * 64 samples)
        sample_points = torch.randn(65536, 3)
        
        # This should work but will fail with dimension mismatch
        try:
            densities, colors = model(sample_points)
            assert densities.shape == (65536, 1)
            assert colors.shape == (65536, 3)
        except RuntimeError as e:
            pytest.fail(f"Dimension mismatch error: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])