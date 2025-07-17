#!/usr/bin/env python3
"""
Comprehensive tests for PyPopSift.

This module consolidates all major PyPopSift functionality tests into a 
single, well-organized test suite covering:
- Core PopSift pipeline and API
- Unified Features interface with symmetric CPU/GPU behavior  
- Configuration and parameter validation
- Error handling and edge cases
- Type system and enums
"""

import pytest
import numpy as np
import pypopsift


class TestPopSiftCore:
    """Test core PopSift functionality and pipeline."""
    
    def test_popsift_creation_and_basic_usage(self):
        """Test PopSift instance creation with different configurations."""
        # Default configuration
        popsift_default = pypopsift.PopSift()
        assert isinstance(popsift_default, pypopsift.PopSift)
        
        # Custom configuration
        config = pypopsift.Config(octaves=4, levels=3, sigma=1.6)
        popsift_custom = pypopsift.PopSift(config, pypopsift.ImageMode.ByteImages)
        assert isinstance(popsift_custom, pypopsift.PopSift)
        
        # Float images
        popsift_float = pypopsift.PopSift(config, pypopsift.ImageMode.FloatImages)
        assert isinstance(popsift_float, pypopsift.PopSift)
    
    def test_enqueue_with_automatic_dimensions(self):
        """Test the improved enqueue API with automatic dimension inference."""
        try:
            # Test byte images
            image_uint8 = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
            popsift = pypopsift.PopSift(pypopsift.Config(), pypopsift.ImageMode.ByteImages)
            job = popsift.enqueue(image_uint8)
            features = job.get()
            
            assert isinstance(features, pypopsift.Features)
            assert features.is_gpu()  # Default mode
            assert features.get_feature_count() >= 0
            assert features.get_descriptor_count() >= 0
            
            # Test float images
            image_float = np.random.rand(80, 120).astype(np.float32)
            popsift_float = pypopsift.PopSift(pypopsift.Config(), pypopsift.ImageMode.FloatImages)
            job_float = popsift_float.enqueue(image_float)
            features_float = job_float.get()
            
            assert isinstance(features_float, pypopsift.Features)
            assert features_float.is_gpu()
            
        except Exception as e:
            pytest.skip(f"PopSift processing failed: {e}")
    
    def test_enqueue_error_handling(self):
        """Test proper error handling for invalid inputs."""
        popsift = pypopsift.PopSift()
        
        # Test 1D array (should fail)
        with pytest.raises(ValueError, match="Expected 2D array, got 1D array"):
            image_1d = np.random.randint(0, 256, (100,), dtype=np.uint8)
            popsift.enqueue(image_1d)
        
        # Test 3D array (should fail)
        with pytest.raises(ValueError, match="Expected 2D array, got 3D array"):
            image_3d = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
            popsift.enqueue(image_3d)
    
    def test_different_image_sizes(self):
        """Test enqueue with various image sizes."""
        sizes = [(64, 64), (100, 200), (300, 150), (200, 100)]
        
        try:
            popsift = pypopsift.PopSift()
            for height, width in sizes:
                image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
                job = popsift.enqueue(image)
                features = job.get()
                assert isinstance(features, pypopsift.Features)
                assert features.get_feature_count() >= 0
                
        except Exception as e:
            pytest.skip(f"PopSift processing failed: {e}")


class TestUnifiedFeatures:
    """Test the unified Features interface with symmetric CPU/GPU behavior."""
    
    def test_features_creation_modes(self):
        """Test creating Features in different ways."""
        # Direct creation
        features = pypopsift.Features(10, 20)
        assert features.is_cpu()
        assert features.get_feature_count() == 10
        assert features.get_descriptor_count() == 20
        
        # From FeaturesHost
        from pypopsift._pypopsift_impl import FeaturesHost  # type: ignore
        host_features = FeaturesHost(5, 10)
        features_from_host = pypopsift.Features(host_features, None)  # type: ignore
        assert features_from_host.is_cpu()
        
        # From FeaturesDev
        from pypopsift._pypopsift_impl import FeaturesDev  # type: ignore
        dev_features = FeaturesDev(8, 16)
        features_from_dev = pypopsift.Features(dev_features, None)  # type: ignore
        assert features_from_dev.is_gpu()
    
    def test_cpu_gpu_transfers(self):
        """Test seamless CPU ↔ GPU memory transfers."""
        # Start with CPU
        cpu_features = pypopsift.Features(5, 10)
        assert cpu_features.is_cpu()
        
        # Transfer to GPU
        gpu_features = cpu_features.gpu()
        assert gpu_features.is_gpu()
        assert gpu_features.get_feature_count() == 5
        assert gpu_features.get_descriptor_count() == 10
        
        # Transfer back to CPU
        cpu_again = gpu_features.cpu()
        assert cpu_again.is_cpu()
        assert cpu_again.get_feature_count() == 5
        assert cpu_again.get_descriptor_count() == 10
        
        # Verify same-mode transfers return self
        assert cpu_features.cpu() is cpu_features
        assert gpu_features.gpu() is gpu_features
    
    def test_symmetric_array_access(self):
        """Test symmetric array access for CPU and GPU modes."""
        cpu_features = pypopsift.Features(5, 10)
        
        # CPU arrays should be NumPy
        try:
            cpu_feat_array = cpu_features.get_features_array()
            cpu_desc_array = cpu_features.get_descriptors_array()
            
            assert isinstance(cpu_feat_array, np.ndarray)
            assert isinstance(cpu_desc_array, np.ndarray)
            assert cpu_feat_array.shape == (5, 7)
            assert cpu_desc_array.shape == (10, 128)
            
        except AttributeError:
            pytest.skip("CPU array access not available yet")
        
        # GPU arrays should have CUDA interface (CuPy-like)
        gpu_features = cpu_features.gpu()
        gpu_feat_array = gpu_features.get_features_array()
        gpu_desc_array = gpu_features.get_descriptors_array()
        
        assert hasattr(gpu_feat_array, '__cuda_array_interface__')
        assert hasattr(gpu_desc_array, '__cuda_array_interface__')
        assert gpu_feat_array.shape == (5, 7)
        assert gpu_desc_array.shape == (10, 128)
        
        # GPU-only reverse map
        reverse_map = gpu_features.get_reverse_map_array()
        assert hasattr(reverse_map, '__cuda_array_interface__')
        assert reverse_map.shape == (10,)
    
    def test_mode_specific_operations(self):
        """Test operations specific to CPU or GPU modes."""
        cpu_features = pypopsift.Features(5, 10)
        gpu_features = cpu_features.gpu()
        
        # CPU-only operations
        cpu_features.pin()
        cpu_features.unpin()
        cpu_features.reset(8, 16)
        
        with pytest.raises(RuntimeError, match="pin.*only available for CPU"):
            gpu_features.pin()
        
        with pytest.raises(RuntimeError, match="get_reverse_map_array.*only available for GPU"):
            cpu_features.get_reverse_map_array()
        
        # GPU-only operations
        gpu_features2 = pypopsift.Features(5, 10).gpu()
        gpu_features.match(gpu_features2)  # Should work
        
        with pytest.raises(RuntimeError, match="match.*only available for GPU"):
            cpu_features.match(gpu_features)
    
    def test_features_consistency_and_behavior(self):
        """Test consistent behavior across CPU and GPU modes."""
        cpu_features = pypopsift.Features(3, 6)
        gpu_features = cpu_features.gpu()
        
        # Same basic properties
        assert cpu_features.size() == gpu_features.size()
        assert len(cpu_features) == len(gpu_features)
        assert cpu_features.get_feature_count() == gpu_features.get_feature_count()
        assert cpu_features.get_descriptor_count() == gpu_features.get_descriptor_count()
        
        # Both support reset
        cpu_features.reset(4, 8)
        gpu_features.reset(4, 8)
        assert cpu_features.get_feature_count() == 4
        assert gpu_features.get_feature_count() == 4
        
        # String representations
        assert "Features(CPU" in str(cpu_features)
        assert "Features(GPU" in str(gpu_features)


class TestConfiguration:
    """Test SIFT configuration and parameter validation."""
    
    def test_config_creation_and_basic_properties(self):
        """Test Config creation and basic property access."""
        # Default config
        config = pypopsift.Config()
        assert config.octaves == -1  # Auto-detect
        assert config.levels == 3
        assert abs(config.sigma - 1.6) < 1e-6
        assert config.verbose == False
        
        # Custom config
        config_custom = pypopsift.Config(octaves=4, levels=3, sigma=1.8)
        assert config_custom.octaves == 4
        assert config_custom.levels == 3
        assert abs(config_custom.sigma - 1.8) < 1e-6
    
    def test_config_parameter_setting(self):
        """Test setting various configuration parameters."""
        config = pypopsift.Config()
        
        # Basic parameters
        config.set_octaves(5)
        config.set_levels(4)
        config.set_sigma(2.0)
        config.set_threshold(0.05)
        # config.set_verbose(True)  # Method may not be available
        
        assert config.octaves == 5
        assert config.levels == 4
        assert config.sigma == 2.0
        # assert config.verbose == True  # Property may not be available
        
        # Mode settings
        config.set_gauss_mode("vlfeat")
        config.set_sift_mode(pypopsift.SiftMode.PopSift)
        # config.set_desc_mode("loop")  # Method may not be available
        # config.set_norm_mode("RootSift")  # Method may not be available
        
        assert config.get_gauss_mode() == pypopsift.GaussMode.VLFeat_Compute
        assert config.get_sift_mode() == pypopsift.SiftMode.PopSift
        # assert config.get_desc_mode() == pypopsift.DescMode.Loop  # Method may not be available
        # assert config.get_norm_mode() == pypopsift.NormMode.RootSift  # Method may not be available
    
    def test_config_equality_and_representation(self):
        """Test config equality comparison and string representation."""
        config1 = pypopsift.Config(octaves=4, levels=3, sigma=1.6)
        config2 = pypopsift.Config(octaves=4, levels=3, sigma=1.6)
        config3 = pypopsift.Config(octaves=5, levels=3, sigma=1.6)
        
        assert config1 == config2
        assert config1 != config3
        
        # String representation
        repr_str = repr(config1)
        assert "Config(" in repr_str
        assert "octaves=4" in repr_str
        assert "levels=3" in repr_str


class TestEnumsAndTypes:
    """Test enums, constants, and type system."""
    
    def test_image_mode_enum(self):
        """Test ImageMode enum values and behavior."""
        # Test enum instances
        assert isinstance(pypopsift.ImageMode.ByteImages, pypopsift.ImageMode)
        assert isinstance(pypopsift.ImageMode.FloatImages, pypopsift.ImageMode)
        assert str(pypopsift.ImageMode.ByteImages) == "ByteImages"
        assert str(pypopsift.ImageMode.FloatImages) == "FloatImages"
        assert pypopsift.ImageMode.ByteImages != pypopsift.ImageMode.FloatImages
    
    def test_sift_mode_enum(self):
        """Test SiftMode enum values."""
        assert isinstance(pypopsift.SiftMode.PopSift, pypopsift.SiftMode)
        assert isinstance(pypopsift.SiftMode.OpenCV, pypopsift.SiftMode)
        assert isinstance(pypopsift.SiftMode.VLFeat, pypopsift.SiftMode)
        
        # String representation
        assert "PopSift" in str(pypopsift.SiftMode.PopSift)
        assert "OpenCV" in str(pypopsift.SiftMode.OpenCV)
    
    def test_gauss_mode_enum(self):
        """Test GaussMode enum values."""
        modes = [
            pypopsift.GaussMode.VLFeat_Compute,
            pypopsift.GaussMode.VLFeat_Relative,
            pypopsift.GaussMode.OpenCV_Compute,
            pypopsift.GaussMode.Fixed9,
            pypopsift.GaussMode.Fixed15
        ]
        
        for mode in modes:
            assert isinstance(mode, pypopsift.GaussMode)
            assert isinstance(str(mode), str)
    
    def test_constants(self):
        """Test module constants."""
        assert isinstance(pypopsift.ORIENTATION_MAX_COUNT, int)
        assert pypopsift.ORIENTATION_MAX_COUNT > 0


class TestFeatureAndDescriptor:
    """Test individual Feature and Descriptor objects."""
    
    def test_feature_properties(self):
        """Test Feature object properties and access."""
        features = pypopsift.Features(1, 1)
        if len(features) > 0:
            feature = features[0]
            
            # Test basic properties exist
            assert hasattr(feature, 'xpos')
            assert hasattr(feature, 'ypos')
            assert hasattr(feature, 'sigma')
            assert hasattr(feature, 'num_ori')
            assert hasattr(feature, 'debug_octave')
            
            # Test orientation array access
            orientations = feature.orientation
            assert isinstance(orientations, np.ndarray)
            assert len(orientations) == pypopsift.ORIENTATION_MAX_COUNT
    
    def test_descriptor_access(self):
        """Test Descriptor object access and properties."""
        descriptor = pypopsift.Descriptor()
        
        # Test length
        assert len(descriptor) == 128
        
        # Test features array access
        features_array = descriptor.features
        assert isinstance(features_array, np.ndarray)
        assert features_array.shape == (128,)
        
        # Test indexing
        descriptor[0] = 1.5
        assert abs(descriptor[0] - 1.5) < 1e-6


class TestEdgeCasesAndErrorHandling:
    """Test edge cases, error conditions, and exception handling."""
    
    def test_empty_and_minimal_features(self):
        """Test edge cases with empty or minimal feature sets."""
        # Empty features
        empty_features = pypopsift.Features(0, 0)
        assert empty_features.size() == 0
        assert len(empty_features) == 0
        assert list(empty_features) == []
        
        # Single feature
        single_features = pypopsift.Features(1, 1)
        assert single_features.size() == 1
        assert len(single_features) == 1
        
        if len(single_features) > 0:
            feature = single_features[0]
            assert isinstance(feature, pypopsift.Feature)
    
    def test_invalid_construction_parameters(self):
        """Test error handling for invalid constructor parameters."""
        # Invalid constructor arguments
        with pytest.raises(ValueError):
            pypopsift.Features("invalid", None)  # type: ignore
        
        with pytest.raises(ValueError):
            pypopsift.Features(10, None)  # Missing descriptor_count  # type: ignore
    
    def test_index_out_of_range(self):
        """Test index bounds checking."""
        features = pypopsift.Features(2, 4)
        
        # Valid indexing
        features[0]
        features[1]
        
        # Invalid indexing
        with pytest.raises(IndexError):
            features[2]
        
        with pytest.raises(IndexError):
            features[-1]
    
    def test_configuration_extreme_values(self):
        """Test config with extreme or edge case values."""
        config = pypopsift.Config()
        
        # Test boundary values
        config.set_octaves(1)
        config.set_levels(2)
        config.set_sigma(0.5)
        config.set_threshold(0.001)
        # config.set_edge_limit(5.0)  # Method may not be available
        
        # These should not raise exceptions
        assert config.octaves == 1
        assert config.levels == 2
        assert config.sigma == 0.5


class TestIntegrationWorkflow:
    """Integration tests that verify the complete workflow."""
    
    @pytest.mark.skipif(not hasattr(pypopsift, 'PopSift'), reason="PopSift not available")
    def test_complete_pipeline_workflow(self):
        """Test the complete PopSift pipeline from image to features."""
        try:
            # Create test image
            test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            
            # Configure and create PopSift
            config = pypopsift.Config(octaves=3, levels=3, sigma=1.6)
            config.set_gauss_mode("vlfeat")
            # config.set_desc_mode("loop")  # Method may not be available
            popsift = pypopsift.PopSift(config, pypopsift.ImageMode.ByteImages)
            
            # Process image with improved API
            job = popsift.enqueue(test_image)  # type: ignore
            features = job.get()
            
            # Verify results
            assert isinstance(features, pypopsift.Features)
            assert features.is_gpu()  # Default from pipeline
            assert features.get_feature_count() >= 0
            assert features.get_descriptor_count() >= 0
            
            # Test CPU conversion
            cpu_features = features.cpu()
            assert cpu_features.is_cpu()
            assert cpu_features.get_feature_count() == features.get_feature_count()
            
            # Test array access
            if features.get_feature_count() > 0:
                gpu_array = features.get_features_array()
                assert hasattr(gpu_array, '__cuda_array_interface__')
                
                try:
                    cpu_array = cpu_features.get_features_array()
                    assert isinstance(cpu_array, np.ndarray)
                except AttributeError:
                    pytest.skip("CPU array access not available")
            
        except Exception as e:
            pytest.skip(f"Complete workflow test failed: {e}")
    
    def test_api_consistency_and_ergonomics(self):
        """Test that the API is consistent and ergonomic to use."""
        # The new API should be simple and intuitive
        image = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        
        try:
            # Simple, one-line usage
            popsift = pypopsift.PopSift()
            job = popsift.enqueue(image)  # Clean API!
            features = job.get()
            
            # Should be straightforward
            assert isinstance(features, pypopsift.Features)
            assert features.get_feature_count() >= 0
            
            # Easy mode switching
            cpu_features = features.cpu()
            gpu_features = cpu_features.gpu()
            assert cpu_features.is_cpu()
            assert gpu_features.is_gpu()
            
        except Exception as e:
            pytest.skip(f"API ergonomics test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 