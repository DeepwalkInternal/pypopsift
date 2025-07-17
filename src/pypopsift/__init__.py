"""
PyPopSift: Python bindings for PopSift SIFT feature extraction

This module provides Python access to PopSift's real-time SIFT implementation
with comprehensive input validation, automatic dimension inference, and
symmetric CPU/GPU feature interfaces.
"""

import numpy as np
from . import _pypopsift_impl  # type: ignore
from .features import Features

# Re-export all the basic types and enums from the C++ module
from ._pypopsift_impl import (  # type: ignore
    # Basic classes (we'll wrap some of these)
    Descriptor, Feature, FeaturesHost, FeaturesDev, SiftJob,
    
    # Enums
    ImageMode, AllocTest, GaussMode, SiftMode, LogMode, ScalingMode, 
    DescMode, NormMode, GridFilterMode,
    
    # Constants
    ORIENTATION_MAX_COUNT,
    
    # Exceptions
    ConfigError, InvalidEnumError, ParameterRangeError, MemoryError,
    CudaError, ImageError, UnsupportedOperationError, LogicError
)


class Config:
    """
    Enhanced SIFT configuration with integrated ImageMode support.
    
    This wrapper extends the base Config class to include ImageMode as a 
    configuration parameter (where it logically belongs) and provides
    a cleaner API design.
    
    Examples:
        Basic usage:
        
        >>> config = Config()
        >>> config.set_image_mode(ImageMode.ByteImages)
        >>> config.set_sigma(1.6)
        
        Advanced configuration:
        
        >>> config = Config(octaves=4, levels=3, sigma=1.8)
        >>> config.set_image_mode(ImageMode.FloatImages)
        >>> config.set_gauss_mode("vlfeat")
    """
    
    def __init__(self, octaves=-1, levels=3, sigma=1.6):
        """
        Create a new SIFT configuration.
        
        Args:
            octaves: Number of octaves (-1 for auto-detection)
            levels: Number of levels per octave
            sigma: Initial smoothing sigma value
        """
        self._config = _pypopsift_impl.Config(octaves, levels, sigma)  # type: ignore
        self._image_mode = ImageMode.ByteImages  # Default to uint8 images
        
    def set_image_mode(self, mode):
        """
        Set the image mode for input validation.
        
        Args:
            mode: ImageMode.ByteImages for uint8 or ImageMode.FloatImages for float32
        """
        if not isinstance(mode, ImageMode):
            raise TypeError(f"Expected ImageMode, got {type(mode)}")
        self._image_mode = mode
        
    def get_image_mode(self):
        """Get the current image mode."""
        return self._image_mode
        
    # Delegate all other methods to the underlying C++ Config
    def __getattr__(self, name):
        return getattr(self._config, name)
        
    def __setattr__(self, name, value):
        # Handle our custom attributes
        if name.startswith('_') or name in ['set_image_mode', 'get_image_mode']:
            super().__setattr__(name, value)
        else:
            # Delegate to underlying config
            if hasattr(self, '_config'):
                setattr(self._config, name, value)
            else:
                super().__setattr__(name, value)
                
    def __repr__(self):
        return f"Config(octaves={self.octaves}, levels={self.levels}, sigma={self.sigma:.3f}, image_mode={self._image_mode})"
    
    def __eq__(self, other):
        """Compare Config objects for equality."""
        if not isinstance(other, Config):
            return False
        return (self._config == other._config and 
                self._image_mode == other._image_mode)
    
    def __ne__(self, other):
        """Compare Config objects for inequality."""
        return not self.__eq__(other)


class PopSift:
    """
    Enhanced PopSift pipeline with input validation and automatic uint8 support.
    
    This wrapper provides:
    - Automatic uint8 to float32 conversion when needed
    - Input validation for image types and value ranges
    - Clean API with ImageMode integrated into Config
    - Comprehensive error messages
    
    Examples:
        Basic usage:
        
        >>> config = Config()
        >>> config.set_image_mode(ImageMode.ByteImages)
        >>> popsift = PopSift(config)
        >>> job = popsift.enqueue(uint8_image)  # Automatic validation and processing
        
        Advanced usage:
        
        >>> config = Config(octaves=4, levels=3, sigma=1.6)
        >>> config.set_image_mode(ImageMode.FloatImages)
        >>> popsift = PopSift(config)
        >>> job = popsift.enqueue(float32_image)  # Validates range [0.0, 1.0]
    """
    
    def __init__(self, config=None, image_mode=None, device=0):
        """
        Create a PopSift pipeline.
        
        Args:
            config: Enhanced Config object with ImageMode, or legacy _pypopsift_impl.Config
            image_mode: Legacy ImageMode parameter (for backward compatibility)
            device: CUDA device ID to use
        """
        # Handle backward compatibility
        if config is None:
            # Default case: create default config
            config = Config()
        elif hasattr(config, '_config'):
            # Enhanced Config object
            if not isinstance(config, Config):
                raise TypeError(f"Expected Config object, got {type(config)}")
        else:
            # Legacy _pypopsift_impl.Config - wrap it
            if image_mode is None:
                image_mode = ImageMode.ByteImages
            legacy_config = config
            config = Config()
            config._config = legacy_config  # type: ignore
            config.set_image_mode(image_mode)
            
        self.config = config
        self.device = device
        
        # Create the underlying C++ PopSift instance
        # Use the image mode from config to determine C++ ImageMode
        cpp_image_mode = config.get_image_mode()
        self._popsift = _pypopsift_impl.PopSift(config._config, cpp_image_mode, device)  # type: ignore
        
    def enqueue(self, image_data):
        """
        Enqueue an image for SIFT feature extraction with automatic validation.
        
        Args:
            image_data: 2D numpy array with shape (height, width)
                       - uint8: values 0-255 (requires ImageMode.ByteImages)
                       - float32: values 0.0-1.0 (requires ImageMode.FloatImages)
                       
        Returns:
            SiftJob: Job object for tracking the processing task
            
        Raises:
            ValueError: If image validation fails
            TypeError: If image is not a numpy array or wrong dtype
        """
        # Basic input validation
        if not isinstance(image_data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image_data)}")
            
        if image_data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {image_data.ndim}D array")
            
        if image_data.size == 0:
            raise ValueError("Cannot process empty image")
            
        # Type and mode validation with automatic conversion
        configured_mode = self.config.get_image_mode()
        
        if image_data.dtype == np.uint8:
            # uint8 image processing
            if configured_mode != ImageMode.ByteImages:
                raise ValueError(
                    f"uint8 images require ImageMode.ByteImages, "
                    f"but config is set to {configured_mode}"
                )
            
            # Values 0-255 are automatically valid for uint8
            # Convert to float32 as workaround for C++ binding issue
            processed_image = image_data.astype(np.float32) / 255.0
            
            # Create a temporary PopSift instance for float processing
            temp_popsift = _pypopsift_impl.PopSift(  # type: ignore
                self.config._config, 
                ImageMode.FloatImages,  # Temporary mode for processing
                self.device
            )
            return temp_popsift.enqueue(processed_image)
            
        elif image_data.dtype == np.float32:
            # float32 image processing
            if configured_mode != ImageMode.FloatImages:
                raise ValueError(
                    f"float32 images require ImageMode.FloatImages, "
                    f"but config is set to {configured_mode}"
                )
                
            # Validate value range
            if np.any(image_data < 0) or np.any(image_data > 1):
                min_val, max_val = float(np.min(image_data)), float(np.max(image_data))
                raise ValueError(
                    f"float32 images must have values in range [0.0, 1.0], "
                    f"got range [{min_val:.3f}, {max_val:.3f}]"
                )
                
            # Use the existing PopSift instance (already configured for FloatImages)
            return self._popsift.enqueue(image_data)
            
        else:
            raise TypeError(
                f"Unsupported image dtype: {image_data.dtype}. "
                f"Supported types: uint8 (0-255) or float32 (0.0-1.0)"
            )
    
    def configure(self, config, force=False):
        """Configure the pipeline with new parameters."""
        return self._popsift.configure(config._config, force)
        
    def uninit(self):
        """Release all allocated resources."""
        return self._popsift.uninit()
        
    def test_texture_fit(self, width, height):
        """Test if image dimensions are supported."""
        return self._popsift.test_texture_fit(width, height)
        
    def test_texture_fit_error_string(self, err, width, height):
        """Get error message for texture fit test results."""
        return self._popsift.test_texture_fit_error_string(err, width, height)
        
    def __repr__(self):
        return f"PopSift(config={self.config}, device={self.device})"


# Update the public API
__all__ = [
    # Enhanced classes
    'Config', 'PopSift', 'Features',
    
    # Original classes  
    'Descriptor', 'Feature', 'FeaturesHost', 'FeaturesDev', 'SiftJob',
    
    # Enums
    'ImageMode', 'AllocTest', 'GaussMode', 'SiftMode', 'LogMode', 'ScalingMode',
    'DescMode', 'NormMode', 'GridFilterMode',
    
    # Constants
    'ORIENTATION_MAX_COUNT',
    
    # Exceptions
    'ConfigError', 'InvalidEnumError', 'ParameterRangeError', 'MemoryError',
    'CudaError', 'ImageError', 'UnsupportedOperationError', 'LogicError'
]
