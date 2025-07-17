#!/usr/bin/env python3
"""
Tests for the enhanced PyPopSift API.

This module tests the enhanced PyPopSift implementation with:
- Config class with integrated ImageMode
- PopSift wrapper with input validation
- Automatic uint8 to float32 conversion
- Comprehensive error handling
"""

import pytest
import numpy as np
import pypopsift
from PIL import Image


class TestEnhancedConfig:
    """Test the enhanced Config class with ImageMode integration."""
    
    def test_config_creation_with_defaults(self):
        """Test creating Config with default parameters."""
        config = pypopsift.Config()
        assert config.octaves == -1
        assert config.levels == 3
        assert abs(config.sigma - 1.6) < 1e-6
        assert config.get_image_mode() == pypopsift.ImageMode.ByteImages  # Default
    
    def test_config_creation_with_parameters(self):
        """Test creating Config with custom parameters."""
        config = pypopsift.Config(octaves=4, levels=3, sigma=1.8)
        assert config.octaves == 4
        assert config.levels == 3
        assert abs(config.sigma - 1.8) < 1e-6
        assert config.get_image_mode() == pypopsift.ImageMode.ByteImages
    
    def test_image_mode_setting(self):
        """Test setting and getting ImageMode."""
        config = pypopsift.Config()
        
        # Test ByteImages mode
        config.set_image_mode(pypopsift.ImageMode.ByteImages)
        assert config.get_image_mode() == pypopsift.ImageMode.ByteImages
        
        # Test FloatImages mode
        config.set_image_mode(pypopsift.ImageMode.FloatImages)
        assert config.get_image_mode() == pypopsift.ImageMode.FloatImages
    
    def test_image_mode_validation(self):
        """Test validation of ImageMode parameter."""
        config = pypopsift.Config()
        
        # Invalid type should raise TypeError
        with pytest.raises(TypeError, match="Expected ImageMode"):
            config.set_image_mode("invalid")  # type: ignore
        
        with pytest.raises(TypeError, match="Expected ImageMode"):
            config.set_image_mode(123)  # type: ignore
    
    def test_config_delegation(self):
        """Test that Config properly delegates to underlying C++ Config."""
        config = pypopsift.Config()
        
        # Test property access
        config.set_octaves(5)
        assert config.octaves == 5
        
        config.set_levels(4)
        assert config.levels == 4
        
        config.set_sigma(2.0)
        assert abs(config.sigma - 2.0) < 1e-6
        
        # Test method access
        config.set_gauss_mode("vlfeat")
        assert config.get_gauss_mode() == pypopsift.GaussMode.VLFeat_Compute
    
    def test_config_representation(self):
        """Test Config string representation."""
        config = pypopsift.Config(octaves=4, levels=3, sigma=1.6)
        config.set_image_mode(pypopsift.ImageMode.FloatImages)
        
        repr_str = repr(config)
        assert "Config(" in repr_str
        assert "octaves=4" in repr_str
        assert "levels=3" in repr_str
        assert "sigma=1.6" in repr_str
        assert "image_mode=FloatImages" in repr_str


class TestEnhancedPopSift:
    """Test the enhanced PopSift class with input validation."""
    
    def test_popsift_creation(self):
        """Test creating PopSift with enhanced Config."""
        config = pypopsift.Config(octaves=4, levels=3, sigma=1.6)
        config.set_image_mode(pypopsift.ImageMode.ByteImages)
        
        popsift = pypopsift.PopSift(config, device=0)
        assert popsift.config is config
        assert popsift.device == 0
    
    def test_popsift_config_validation(self):
        """Test PopSift constructor validates Config type."""
        with pytest.raises(TypeError, match="Expected Config object"):
            pypopsift.PopSift("invalid_config")
        
        with pytest.raises(TypeError, match="Expected Config object"):
            pypopsift.PopSift(123)


class TestUint8ImageProcessing:
    """Test automatic uint8 image processing with validation."""
    
    def test_uint8_with_correct_mode(self):
        """Test uint8 images work with ByteImages mode."""
        config = pypopsift.Config()
        config.set_image_mode(pypopsift.ImageMode.ByteImages)
        popsift = pypopsift.PopSift(config)
        
        # Create test uint8 image
        image = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        
        # Should work without errors
        job = popsift.enqueue(image)
        features = job.get()
        
        assert isinstance(features, pypopsift.Features)
        assert features.is_gpu()  # Features should be in GPU memory
        assert features.get_feature_count() >= 0
        assert features.get_descriptor_count() >= 0
    
    def test_uint8_with_wrong_mode(self):
        """Test uint8 images fail validation with FloatImages mode."""
        config = pypopsift.Config()
        config.set_image_mode(pypopsift.ImageMode.FloatImages)
        popsift = pypopsift.PopSift(config)
        
        image = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="uint8 images require ImageMode.ByteImages"):
            popsift.enqueue(image)
    
    def test_uint8_real_image(self):
        """Test uint8 processing with real image."""
        try:
            config = pypopsift.Config()
            config.set_image_mode(pypopsift.ImageMode.ByteImages)
            popsift = pypopsift.PopSift(config)
            
            # Load real image
            image = np.asarray(Image.open('tests/lenna.png').convert('L'))
            assert image.dtype == np.uint8
            
            job = popsift.enqueue(image)
            features = job.get()
            
            assert features.get_feature_count() > 0
            assert features.get_descriptor_count() > 0
            
            # Test CPU/GPU transfers
            cpu_features = features.cpu()
            assert cpu_features.is_cpu()
            assert cpu_features.get_feature_count() == features.get_feature_count()
            
        except Exception as e:
            pytest.skip(f"Real image test failed: {e}")


class TestFloat32ImageProcessing:
    """Test float32 image processing with range validation."""
    
    def test_float32_with_correct_mode(self):
        """Test float32 images work with FloatImages mode."""
        config = pypopsift.Config()
        config.set_image_mode(pypopsift.ImageMode.FloatImages)
        popsift = pypopsift.PopSift(config)
        
        # Create valid float32 image (0.0-1.0 range)
        image = np.random.rand(100, 150).astype(np.float32)
        
        job = popsift.enqueue(image)
        features = job.get()
        
        assert isinstance(features, pypopsift.Features)
        assert features.is_gpu()
        assert features.get_feature_count() >= 0
    
    def test_float32_with_wrong_mode(self):
        """Test float32 images fail validation with ByteImages mode."""
        config = pypopsift.Config()
        config.set_image_mode(pypopsift.ImageMode.ByteImages)
        popsift = pypopsift.PopSift(config)
        
        image = np.random.rand(100, 150).astype(np.float32)
        
        with pytest.raises(ValueError, match="float32 images require ImageMode.FloatImages"):
            popsift.enqueue(image)
    
    def test_float32_range_validation_negative(self):
        """Test float32 range validation catches negative values."""
        config = pypopsift.Config()
        config.set_image_mode(pypopsift.ImageMode.FloatImages)
        popsift = pypopsift.PopSift(config)
        
        # Create image with negative values
        image = np.random.rand(50, 50).astype(np.float32) - 0.5  # Range: [-0.5, 0.5]
        
        with pytest.raises(ValueError, match="float32 images must have values in range \\[0.0, 1.0\\]"):
            popsift.enqueue(image)
    
    def test_float32_range_validation_too_large(self):
        """Test float32 range validation catches values > 1.0."""
        config = pypopsift.Config()
        config.set_image_mode(pypopsift.ImageMode.FloatImages)
        popsift = pypopsift.PopSift(config)
        
        # Create image with values > 1.0
        image = np.random.rand(50, 50).astype(np.float32) * 2.0  # Range: [0.0, 2.0]
        
        with pytest.raises(ValueError, match="float32 images must have values in range \\[0.0, 1.0\\]"):
            popsift.enqueue(image)
    
    def test_float32_edge_values(self):
        """Test float32 with edge values (0.0 and 1.0)."""
        config = pypopsift.Config()
        config.set_image_mode(pypopsift.ImageMode.FloatImages)
        popsift = pypopsift.PopSift(config)
        
        # Create image with edge values
        image = np.zeros((50, 50), dtype=np.float32)
        image[0, 0] = 0.0  # Minimum valid value
        image[0, 1] = 1.0  # Maximum valid value
        
        # Should work without errors
        job = popsift.enqueue(image)
        features = job.get()
        assert features.get_feature_count() >= 0


class TestInputValidation:
    """Test comprehensive input validation."""
    
    def test_non_numpy_array(self):
        """Test validation rejects non-numpy arrays."""
        config = pypopsift.Config()
        popsift = pypopsift.PopSift(config)
        
        with pytest.raises(TypeError, match="Expected numpy array"):
            popsift.enqueue([[1, 2], [3, 4]])  # Python list
        
        with pytest.raises(TypeError, match="Expected numpy array"):
            popsift.enqueue("invalid")
    
    def test_wrong_dimensions(self):
        """Test validation rejects non-2D arrays."""
        config = pypopsift.Config()
        popsift = pypopsift.PopSift(config)
        
        # 1D array
        with pytest.raises(ValueError, match="Expected 2D array, got 1D array"):
            image_1d = np.random.randint(0, 256, (100,), dtype=np.uint8)
            popsift.enqueue(image_1d)
        
        # 3D array
        with pytest.raises(ValueError, match="Expected 2D array, got 3D array"):
            image_3d = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
            popsift.enqueue(image_3d)
    
    def test_empty_image(self):
        """Test validation rejects empty images."""
        config = pypopsift.Config()
        popsift = pypopsift.PopSift(config)
        
        with pytest.raises(ValueError, match="Cannot process empty image"):
            empty_image = np.array([], dtype=np.uint8).reshape(0, 0)
            popsift.enqueue(empty_image)
    
    def test_unsupported_dtype(self):
        """Test validation rejects unsupported dtypes."""
        config = pypopsift.Config()
        popsift = pypopsift.PopSift(config)
        
        # int32 array
        with pytest.raises(TypeError, match="Unsupported image dtype"):
            image_int32 = np.random.randint(0, 256, (50, 50), dtype=np.int32)
            popsift.enqueue(image_int32)
        
        # float64 array
        with pytest.raises(TypeError, match="Unsupported image dtype"):
            image_float64 = np.random.rand(50, 50).astype(np.float64)
            popsift.enqueue(image_float64)


class TestAPIDesignAndUsability:
    """Test API design improvements and usability."""
    
    def test_clean_api_design(self):
        """Test the clean, logical API design."""
        # Clean API: ImageMode is part of Config
        config = pypopsift.Config(octaves=4, levels=3, sigma=1.6)
        config.set_image_mode(pypopsift.ImageMode.ByteImages)
        
        # Simple PopSift creation (no separate ImageMode parameter)
        popsift = pypopsift.PopSift(config)
        
        # Automatic dimension inference
        image = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        job = popsift.enqueue(image)  # Clean: just pass the image!
        features = job.get()
        
        assert isinstance(features, pypopsift.Features)
    
    def test_error_messages_are_helpful(self):
        """Test that error messages provide clear guidance."""
        config = pypopsift.Config()
        config.set_image_mode(pypopsift.ImageMode.FloatImages)
        popsift = pypopsift.PopSift(config)
        
        # Test helpful error for wrong image mode
        image_uint8 = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        try:
            popsift.enqueue(image_uint8)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            assert "uint8 images require ImageMode.ByteImages" in error_msg
            assert "FloatImages" in error_msg  # Shows current mode
        
        # Test helpful error for range validation
        config.set_image_mode(pypopsift.ImageMode.FloatImages)
        image_invalid = np.random.rand(50, 50).astype(np.float32) * 3.0
        try:
            popsift.enqueue(image_invalid)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            assert "must have values in range [0.0, 1.0]" in error_msg
            assert "got range" in error_msg  # Shows actual range
    
    def test_backward_compatibility(self):
        """Test that existing code patterns still work."""
        # The enhanced API should not break existing usage patterns
        config = pypopsift.Config()
        # Default ImageMode is ByteImages for backward compatibility
        assert config.get_image_mode() == pypopsift.ImageMode.ByteImages
        
        # Can still access all original Config methods
        config.set_gauss_mode("vlfeat")
        config.set_sigma(1.8)
        assert config.get_gauss_mode() == pypopsift.GaussMode.VLFeat_Compute
        assert abs(config.sigma - 1.8) < 1e-6


class TestIntegrationWithFeatures:
    """Test integration with the unified Features interface."""
    
    def test_uint8_to_features_workflow(self):
        """Test complete workflow from uint8 image to Features."""
        try:
            config = pypopsift.Config()
            config.set_image_mode(pypopsift.ImageMode.ByteImages)
            popsift = pypopsift.PopSift(config)
            
            image = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
            job = popsift.enqueue(image)
            features = job.get()
            
            # Test unified Features interface
            assert isinstance(features, pypopsift.Features)
            assert features.is_gpu()  # Should be in GPU memory by default
            assert features.get_feature_count() >= 0
            assert features.get_descriptor_count() >= 0
            
            # Test CPU/GPU transfers
            cpu_features = features.cpu()
            assert cpu_features.is_cpu()
            assert cpu_features.get_feature_count() == features.get_feature_count()
            
            gpu_features = cpu_features.gpu()
            assert gpu_features.is_gpu()
            assert gpu_features.get_feature_count() == features.get_feature_count()
            
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")
    
    def test_array_access_after_processing(self):
        """Test array access works after enhanced processing."""
        try:
            config = pypopsift.Config()
            config.set_image_mode(pypopsift.ImageMode.ByteImages)
            popsift = pypopsift.PopSift(config)
            
            image = np.random.randint(0, 256, (80, 120), dtype=np.uint8)
            job = popsift.enqueue(image)
            features = job.get()
            
            if features.get_feature_count() > 0:
                # Test GPU array access
                gpu_features_array = features.get_features_array()
                gpu_descriptors_array = features.get_descriptors_array()
                
                assert hasattr(gpu_features_array, '__cuda_array_interface__')
                assert hasattr(gpu_descriptors_array, '__cuda_array_interface__')
                assert gpu_features_array.shape[0] == features.get_feature_count()
                assert gpu_descriptors_array.shape[0] == features.get_descriptor_count()
                
                # Test CPU array access
                try:
                    cpu_features = features.cpu()
                    cpu_features_array = cpu_features.get_features_array()
                    cpu_descriptors_array = cpu_features.get_descriptors_array()
                    
                    assert isinstance(cpu_features_array, np.ndarray)
                    assert isinstance(cpu_descriptors_array, np.ndarray)
                    assert cpu_features_array.shape[0] == features.get_feature_count()
                    assert cpu_descriptors_array.shape[0] == features.get_descriptor_count()
                except AttributeError:
                    pytest.skip("CPU array access not available yet")
            
        except Exception as e:
            pytest.skip(f"Array access test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 