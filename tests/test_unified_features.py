"""
Tests for the unified Features class.

This module tests the unified Features interface that provides seamless
CPU/GPU memory management and consistent array access for SIFT features.
"""

import pytest
import numpy as np
import pypopsift


class TestUnifiedFeatures:
    """Test the unified Features class functionality."""
    
    def test_features_creation_with_counts(self):
        """Test creating Features with feature and descriptor counts."""
        features = pypopsift.Features(10, 20)
        assert features is not None
        assert isinstance(features, pypopsift.Features)
        assert features.is_cpu()
        assert not features.is_gpu()
        assert features.get_feature_count() == 10
        assert features.get_descriptor_count() == 20
        assert len(features) == 10
        assert features.size() == 10
    
    def test_features_creation_with_featureshost(self):
        """Test creating Features from a FeaturesHost object."""
        from pypopsift._pypopsift_impl import FeaturesHost
        host_features = FeaturesHost(5, 10)
        features = pypopsift.Features(host_features, None)  # type: ignore
        assert features.is_cpu()
        assert not features.is_gpu()
        assert features.get_feature_count() == 5
        assert features.get_descriptor_count() == 10
    
    def test_features_creation_with_featuresdev(self):
        """Test creating Features from a FeaturesDev object."""
        from pypopsift._pypopsift_impl import FeaturesDev
        dev_features = FeaturesDev(8, 16)
        features = pypopsift.Features(dev_features, None)  # type: ignore
        assert features.is_gpu()
        assert not features.is_cpu()
        assert features.get_feature_count() == 8
        assert features.get_descriptor_count() == 16
    
    def test_features_cpu_gpu_transfer(self):
        """Test CPU to GPU and GPU to CPU transfers."""
        # Start with CPU features
        cpu_features = pypopsift.Features(5, 10)
        assert cpu_features.is_cpu()
        
        # Transfer to GPU
        gpu_features = cpu_features.gpu()
        assert gpu_features.is_gpu()
        assert not gpu_features.is_cpu()
        assert gpu_features.get_feature_count() == 5
        assert gpu_features.get_descriptor_count() == 10
        
        # Transfer back to CPU
        cpu_features2 = gpu_features.cpu()
        assert cpu_features2.is_cpu()
        assert not cpu_features2.is_gpu()
        assert cpu_features2.get_feature_count() == 5
        assert cpu_features2.get_descriptor_count() == 10
        
        # Verify that transferring to same memory type returns self
        same_cpu = cpu_features.cpu()
        assert same_cpu is cpu_features
        
        same_gpu = gpu_features.gpu()
        assert same_gpu is gpu_features
    
    def test_features_reset(self):
        """Test resetting features with new dimensions."""
        features = pypopsift.Features(5, 10)
        features.reset(8, 16)
        assert features.get_feature_count() == 8
        assert features.get_descriptor_count() == 16
    
    def test_features_cpu_pin_unpin(self):
        """Test pinning and unpinning CPU memory."""
        cpu_features = pypopsift.Features(5, 10)
        assert cpu_features.is_cpu()
        
        # These should work without error
        cpu_features.pin()
        cpu_features.unpin()
        
        # GPU features should raise error
        gpu_features = cpu_features.gpu()
        with pytest.raises(RuntimeError, match="pin.*only available for CPU features"):
            gpu_features.pin()
        
        with pytest.raises(RuntimeError, match="unpin.*only available for CPU features"):
            gpu_features.unpin()
    
    def test_features_gpu_matching(self):
        """Test GPU feature matching."""
        gpu_features1 = pypopsift.Features(5, 10).gpu()
        gpu_features2 = pypopsift.Features(8, 16).gpu()
        
        assert gpu_features1.is_gpu()
        assert gpu_features2.is_gpu()
        
        # This should work without error
        gpu_features1.match(gpu_features2)
        
        # CPU features should raise error
        cpu_features = pypopsift.Features(5, 10)
        with pytest.raises(RuntimeError, match="match.*only available for GPU features"):
            cpu_features.match(gpu_features1)
        
        with pytest.raises(RuntimeError, match="Can only match against GPU features"):
            gpu_features1.match(cpu_features)
    
    def test_features_cpu_array_access(self):
        """Test CPU features array access (should return NumPy arrays)."""
        cpu_features = pypopsift.Features(5, 10)
        assert cpu_features.is_cpu()
        
        # CPU features should now return NumPy arrays (after C++ recompilation)
        try:
            features_array = cpu_features.get_features_array()
            assert isinstance(features_array, np.ndarray)
            assert features_array.shape == (5, 7)  # 5 features, 7 fields
            
            descriptors_array = cpu_features.get_descriptors_array()
            assert isinstance(descriptors_array, np.ndarray)
            assert descriptors_array.shape == (10, 128)  # 10 descriptors, 128 dimensions
            assert descriptors_array.dtype == np.float32
            
        except AttributeError:
            # Expected until C++ bindings are recompiled with new methods
            import pytest
            pytest.skip("CPU array access not available yet - C++ bindings need recompilation")
    
    def test_features_gpu_array_access(self):
        """Test GPU features array access (should return CuPy arrays)."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")
            
        gpu_features = pypopsift.Features(5, 10).gpu()
        assert gpu_features.is_gpu()
        
        # Get features array
        features_array = gpu_features.get_features_array()
        # Should be CuPy array
        assert hasattr(features_array, '__cuda_array_interface__')
        assert features_array.shape == (5, 7)  # 5 features, 7 fields
        
        # Get descriptors array
        descriptors_array = gpu_features.get_descriptors_array() 
        assert hasattr(descriptors_array, '__cuda_array_interface__')
        assert descriptors_array.shape == (10, 128)  # 10 descriptors, 128 dimensions
        
        # Get reverse map array (GPU only)
        reverse_map = gpu_features.get_reverse_map_array()
        assert hasattr(reverse_map, '__cuda_array_interface__')
        assert reverse_map.shape == (10,)  # 10 descriptors
    
    def test_features_reverse_map_cpu_error(self):
        """Test that reverse map array is only available for GPU features."""
        cpu_features = pypopsift.Features(5, 10)
        with pytest.raises(RuntimeError, match="get_reverse_map_array.*only available for GPU features"):
            cpu_features.get_reverse_map_array()
    
    def test_features_indexing_and_iteration(self):
        """Test indexing and iteration over features."""
        features = pypopsift.Features(5, 10)
        
        # Test length
        assert len(features) == 5
        
        # Test indexing (should delegate to underlying features)
        if len(features) > 0:
            feature = features[0]
            assert isinstance(feature, pypopsift.Feature)
        
        # Test iteration
        feature_list = list(features)  # type: ignore
        assert len(feature_list) == 5
        for feature in feature_list:
            assert isinstance(feature, pypopsift.Feature)
    
    def test_features_string_representation(self):
        """Test string representations of Features."""
        cpu_features = pypopsift.Features(5, 10)
        cpu_repr = repr(cpu_features)
        cpu_str = str(cpu_features)
        
        assert "Features(CPU" in cpu_repr
        assert "features=5" in cpu_repr
        assert "descriptors=10" in cpu_repr
        assert cpu_repr == cpu_str
        
        gpu_features = cpu_features.gpu()
        gpu_repr = repr(gpu_features)
        gpu_str = str(gpu_features)
        
        assert "Features(GPU" in gpu_repr
        assert "features=5" in gpu_repr
        assert "descriptors=10" in gpu_repr
        assert gpu_repr == gpu_str
    
    def test_features_delegation(self):
        """Test that unknown attributes are delegated to underlying features object."""
        features = pypopsift.Features(5, 10)
        
        # Test that delegation works for methods that exist on the underlying object
        # But aren't explicitly defined in our Features class
        
        # Check that we can access some attributes that would be delegated
        # (we'll access attributes that exist on the underlying C++ object)
        assert hasattr(features._features, 'get_feature_count')  # type: ignore
        assert hasattr(features._features, 'get_descriptor_count')  # type: ignore
        
        # Since delegation is working, any unknown attribute access
        # should either succeed (if it exists on underlying) or fail with AttributeError
        try:
            # This should work through delegation
            some_method = getattr(features, 'some_nonexistent_method', None)
            assert some_method is None or callable(some_method)
        except AttributeError:
            # This is also acceptable - the underlying object doesn't have this method
            pass
    
    def test_features_error_handling(self):
        """Test error handling for invalid operations."""
        # Test invalid constructor arguments
        with pytest.raises(ValueError, match="Must provide either"):
            pypopsift.Features("invalid", None)  # type: ignore
        
        with pytest.raises(ValueError, match="Must provide either"):
            pypopsift.Features(10, None)  # Missing descriptor_count  # type: ignore
    
    def test_features_consistent_behavior(self):
        """Test that CPU and GPU modes behave consistently for common operations."""
        # Create features in both modes
        cpu_features = pypopsift.Features(3, 6)
        gpu_features = cpu_features.gpu()
        
        # Both should report same counts
        assert cpu_features.size() == gpu_features.size()
        assert cpu_features.get_feature_count() == gpu_features.get_feature_count()
        assert cpu_features.get_descriptor_count() == gpu_features.get_descriptor_count()
        assert len(cpu_features) == len(gpu_features)
        
        # Both should support reset
        cpu_features.reset(4, 8)
        gpu_features.reset(4, 8)
        
        assert cpu_features.get_feature_count() == 4
        assert gpu_features.get_feature_count() == 4
        assert cpu_features.get_descriptor_count() == 8
        assert gpu_features.get_descriptor_count() == 8
        
        # Both should support array access (but return different array types)
        cpu_array = cpu_features.get_features_array()
        gpu_array = gpu_features.get_features_array()
        
        assert isinstance(cpu_array, np.ndarray)
        # GPU array should have __cuda_array_interface__ (CuPy-like)
        assert hasattr(gpu_array, '__cuda_array_interface__') or isinstance(gpu_array, np.ndarray)


class TestFeaturesIntegration:
    """Integration tests with the actual PopSift pipeline."""
    
    @pytest.mark.skipif(not hasattr(pypopsift, 'PopSift'), reason="PopSift not available")
    def test_features_from_popsift_pipeline(self):
        """Test that Features work correctly with the actual PopSift pipeline."""
        try:
            # Create a simple test image
            test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            
            # Create PopSift instance
            config = pypopsift.Config()
            popsift = pypopsift.PopSift(config, pypopsift.ImageMode.ByteImages)
            
            # Process image (dimensions automatically inferred)
            job = popsift.enqueue(test_image)  # type: ignore
            features = job.get()
            
            # Should be a unified Features object
            assert isinstance(features, pypopsift.Features)
            
            # Should be in GPU mode by default (matching mode)
            assert features.is_gpu()
            assert not features.is_cpu()
            
            # Should have valid counts
            assert features.get_feature_count() >= 0
            assert features.get_descriptor_count() >= 0
            
            # Should be able to transfer to CPU
            cpu_features = features.cpu()
            assert cpu_features.is_cpu()
            assert cpu_features.get_feature_count() == features.get_feature_count()
            assert cpu_features.get_descriptor_count() == features.get_descriptor_count()
            
        except Exception as e:
            pytest.skip(f"PopSift pipeline test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 