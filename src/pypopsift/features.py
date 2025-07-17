"""
Unified Features class for PopSift.

This module provides a unified interface for SIFT features that can be stored
in either CPU (host) or GPU (device) memory, with methods to transfer between them.
"""

from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ._pypopsift_impl import FeaturesHost, FeaturesDev # type: ignore
else:
    try:
        from ._pypopsift_impl import FeaturesHost, FeaturesDev
    except ImportError:
        # For development/testing when the module isn't built yet
        FeaturesHost = None
        FeaturesDev = None


class Features:
    """
    Unified container for SIFT features and descriptors.
    
    Features is a unified interface that can hold features in either CPU (host) or GPU (device) memory.
    It provides methods to transfer data between CPU and GPU memory as needed.
    
    Examples:
        Basic usage:
        
        >>> features = Features(100, 200)  # CPU memory by default
        >>> features_gpu = features.gpu()  # Transfer to GPU
        >>> features_cpu = features_gpu.cpu()  # Transfer back to CPU
        
        From PopSift pipeline:
        
        >>> job = popsift.enqueue(width, height, image_data)
        >>> features = job.get()  # Unified Features object (GPU by default)
        >>> features_cpu = features.cpu()  # Get CPU copy if needed
        
        Array access (symmetric for both CPU and GPU):
        
        >>> features = job.get()  # GPU memory
        >>> features_array = features.get_features_array()  # CuPy array
        >>> descriptors_array = features.get_descriptors_array()  # CuPy array
        
        >>> features_cpu = features.cpu()  # CPU memory
        >>> features_array = features_cpu.get_features_array()  # NumPy array
        >>> descriptors_array = features_cpu.get_descriptors_array()  # NumPy array
        
        GPU matching:
        
        >>> features1 = job1.get()  # GPU memory
        >>> features2 = job2.get()  # GPU memory
        >>> features1.match(features2)  # GPU matching
    """
    
    def __init__(self, features_or_count: Union['FeaturesHost', 'FeaturesDev', int], 
                 descriptor_count: Optional[int] = None):
        """
        Initialize Features.
        
        Args:
            features_or_count: Either a FeaturesHost/FeaturesDev object or pointer, or the number of features
            descriptor_count: Number of descriptors (only used if features_or_count is an int)
        """
        # Accept both objects and pointers (nanobind may return pointers)
        host_types = tuple(t for t in (FeaturesHost,) if t is not None)
        dev_types = tuple(t for t in (FeaturesDev,) if t is not None)
        if FeaturesHost is not None and FeaturesDev is not None:
            if isinstance(features_or_count, host_types + dev_types):
                self._features = features_or_count  # type: ignore
            elif hasattr(features_or_count, '__class__') and (
                features_or_count.__class__.__name__ in ('FeaturesHost', 'FeaturesDev')):
                self._features = features_or_count  # type: ignore
            elif isinstance(features_or_count, int) and descriptor_count is not None:
                # Default to CPU memory
                self._features = FeaturesHost(features_or_count, descriptor_count)  # type: ignore
            else:
                raise ValueError("Must provide either a FeaturesHost/FeaturesDev object or (feature_count, descriptor_count)")
        else:
            raise ImportError("PopSift implementation not available")
    
    @property
    def _is_cpu(self) -> bool:
        """Check if features are in CPU memory."""
        return FeaturesHost is not None and isinstance(self._features, FeaturesHost)
    
    @property
    def _is_gpu(self) -> bool:
        """Check if features are in GPU memory."""
        return FeaturesDev is not None and isinstance(self._features, FeaturesDev)
    
    def _check_initialized(self):
        """Check if features object is properly initialized."""
        if not hasattr(self._features, "size"):
            raise RuntimeError("Features object is not initialized with valid data")

    def cpu(self) -> 'Features':
        """
        Get features in CPU memory.
        
        Returns:
            Features: Features object in CPU memory
            
        Note:
            If already in CPU memory, returns self. If in GPU memory,
            performs device-to-host copy and returns new CPU features.
        """
        if self._is_cpu:
            return self
        elif self._is_gpu:
            self._check_initialized()
            host_features = self._features.to_host()  # type: ignore
            return Features(host_features)
        else:
            raise RuntimeError("Features object is not initialized with valid data (no .cpu() possible)")
    
    def gpu(self) -> 'Features':
        """
        Get features in GPU memory.
        
        Returns:
            Features: Features object in GPU memory
            
        Note:
            If already in GPU memory, returns self. If in CPU memory,
            performs host-to-device copy and returns new GPU features.
        """
        if self._is_gpu:
            return self
        elif self._is_cpu:
            self._check_initialized()
            dev_features = self._features.to_gpu()  # type: ignore
            return Features(dev_features)
        else:
            raise RuntimeError("Features object is not initialized with valid data (no .gpu() possible)")
    
    def is_cpu(self) -> bool:
        """Check if features are in CPU memory."""
        return self._is_cpu
    
    def is_gpu(self) -> bool:
        """Check if features are in GPU memory."""
        return self._is_gpu
    
    def size(self) -> int:
        """Get the number of features."""
        self._check_initialized()
        return self._features.size()  # type: ignore
    
    def get_feature_count(self) -> int:
        """Get the number of features."""
        self._check_initialized()
        return self._features.get_feature_count()  # type: ignore
    
    def get_descriptor_count(self) -> int:
        """Get the number of descriptors."""
        self._check_initialized()
        return self._features.get_descriptor_count()  # type: ignore
    
    def reset(self, feature_count: int, descriptor_count: int):
        """
        Reset the container with new feature and descriptor counts.
        
        Args:
            feature_count: Number of features
            descriptor_count: Number of descriptors
        """
        self._check_initialized()
        self._features.reset(feature_count, descriptor_count)  # type: ignore
    
    def pin(self):
        """
        Pin memory for CUDA operations (CPU features only).
        
        Note:
            This method is only available for CPU features and helps
            optimize host-to-device memory transfers.
        """
        if not self._is_cpu:
            raise RuntimeError("pin() is only available for CPU features")
        self._check_initialized()
        self._features.pin()  # type: ignore
    
    def unpin(self):
        """
        Unpin memory after CUDA operations (CPU features only).
        
        Note:
            This method is only available for CPU features and should
            be called after GPU operations are complete.
        """
        if not self._is_cpu:
            raise RuntimeError("unpin() is only available for CPU features")
        self._check_initialized()
        self._features.unpin()  # type: ignore
    
    def match(self, other: 'Features'):
        """
        Match features against another Features object (GPU features only).
        
        Args:
            other: Another Features object to match against
            
        Note:
            This method is only available for GPU features and performs
            GPU-accelerated feature matching.
        """
        if not self._is_gpu:
            raise RuntimeError("match() is only available for GPU features")
        if not other.is_gpu():
            raise RuntimeError("Can only match against GPU features")
        self._check_initialized()
        self._features.match(other._features)  # type: ignore
    
    def get_features_array(self):
        """
        Get features as an array in memory.
        
        Returns:
            numpy.ndarray or cupy.ndarray: Array view of features in memory.
                         Shape is (num_features, 7) where the 7 fields are:
                         [debug_octave, xpos, ypos, sigma, num_ori, orientation[0], orientation[1]]
        
        Note:
            Returns NumPy array for CPU features, CuPy array for GPU features.
            This returns a view into the memory, not a copy.
            The array is only valid while the Features object exists.
        """
        self._check_initialized()
        return self._features.get_features_array()  # type: ignore
    
    def get_descriptors_array(self):
        """
        Get descriptors as an array in memory.
        
        Returns:
            numpy.ndarray or cupy.ndarray: Array view of descriptors in memory.
                         Shape is (num_descriptors, 128) for SIFT features.
        
        Note:
            Returns NumPy array for CPU features, CuPy array for GPU features.
            This returns a view into the memory, not a copy.
            The array is only valid while the Features object exists.
        """
        self._check_initialized()
        return self._features.get_descriptors_array()  # type: ignore
    
    def get_reverse_map_array(self):
        """
        Get reverse mapping as an array in memory (GPU features only).
        
        Returns:
            cupy.ndarray: CuPy array view of reverse mapping in device memory.
                         Shape is (num_descriptors,) mapping descriptors to features.
        
        Note:
            This method is only available for GPU features.
            This returns a view into the device memory, not a copy.
            The array is only valid while the Features object exists.
        """
        if not self._is_gpu:
            raise RuntimeError("get_reverse_map_array() is only available for GPU features")
        self._check_initialized()
        return self._features.get_reverse_map_array()  # type: ignore
    
    def __len__(self) -> int:
        """Get the number of features."""
        return self.size()
    
    def __getitem__(self, index: int):
        """Get feature by index."""
        self._check_initialized()
        return self._features[index]  # type: ignore
    
    def __iter__(self):
        """Iterate over features."""
        self._check_initialized()
        return iter(self._features)  # type: ignore
    
    def __repr__(self) -> str:
        """String representation."""
        if self._is_cpu:
            return f"Features(CPU, features={self.get_feature_count()}, descriptors={self.get_descriptor_count()})"
        elif self._is_gpu:
            return f"Features(GPU, features={self.get_feature_count()}, descriptors={self.get_descriptor_count()})"
        else:
            return "Features(Unknown)"
    
    def __str__(self) -> str:
        """String representation."""
        return self.__repr__()
    
    # Delegate other methods to the underlying features object
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying features object."""
        # Don't delegate methods that are explicitly implemented in this class
        explicit_methods = {
            'reset', 'pin', 'unpin', 'match', 
            'get_features_array', 'get_descriptors_array', 'get_reverse_map_array',
            'cpu', 'gpu', 'is_cpu', 'is_gpu', 'size', 'get_feature_count', 'get_descriptor_count',
            '__len__', '__getitem__', '__iter__', '__repr__', '__str__'
        }
        if name in explicit_methods:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self._features, name) 