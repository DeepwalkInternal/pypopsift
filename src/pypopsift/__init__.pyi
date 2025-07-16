"""
Type stubs for pypopsift - Python bindings for PopSift SIFT library.

This file provides comprehensive type information for static analysis tools,
IDEs, and type checkers while preserving the rich documentation from the
underlying nanobind C++ implementation.
"""

from typing import Literal, overload
import typing
from collections.abc import Iterator
import cupy

# Constants
ORIENTATION_MAX_COUNT: int

# Exception hierarchy with proper inheritance
class ConfigError(ValueError):
    """Raised when SIFT configuration parameters are invalid."""
    def __init__(self, message: str) -> None: ...

class InvalidEnumError(ValueError):
    """Raised when an invalid enumeration value is provided."""
    ...

class ParameterRangeError(ValueError):
    """Raised when a parameter value is outside its valid range."""
    ...

class MemoryError(Exception):
    """Raised when memory allocation fails in PopSift operations."""
    ...

class CudaError(RuntimeError):
    """Raised when CUDA operations fail."""
    ...

class ImageError(RuntimeError):
    """Raised when image processing operations fail."""
    ...

class UnsupportedOperationError(NotImplementedError):
    """Raised when an unsupported operation is attempted."""
    ...

class LogicError(RuntimeError):
    """Raised when internal logic errors occur."""
    ...

# Enumerations
class GaussMode:
    """Gaussian filtering mode enumeration."""
    VLFeat_Compute: GaussMode
    VLFeat_Relative: GaussMode
    VLFeat_Relative_All: GaussMode
    OpenCV_Compute: GaussMode
    Fixed9: GaussMode
    Fixed15: GaussMode
    def __str__(self) -> str: ...

class SiftMode:
    """SIFT algorithm mode enumeration."""
    PopSift: SiftMode
    OpenCV: SiftMode
    VLFeat: SiftMode
    def __str__(self) -> str: ...

class LogMode:
    """Logging mode enumeration."""
    NONE: LogMode
    All: LogMode
    def __str__(self) -> str: ...

class ScalingMode:
    """Image scaling mode enumeration."""
    ScaleDirect: ScalingMode
    ScaleDefault: ScalingMode
    def __str__(self) -> str: ...

class DescMode:
    """Descriptor computation mode enumeration."""
    Loop: DescMode
    ILoop: DescMode
    Grid: DescMode
    IGrid: DescMode
    NoTile: DescMode
    def __str__(self) -> str: ...

class NormMode:
    """Descriptor normalization mode enumeration."""
    RootSift: NormMode
    Classic: NormMode
    def __str__(self) -> str: ...

class GridFilterMode:
    """Grid filtering mode enumeration."""
    RandomScale: GridFilterMode
    LargestScaleFirst: GridFilterMode
    SmallestScaleFirst: GridFilterMode
    def __str__(self) -> str: ...

class ProcessingMode:
    """Processing mode enumeration."""
    ExtractingMode: ProcessingMode
    MatchingMode: ProcessingMode
    def __str__(self) -> str: ...

class ImageMode:
    """Image mode enumeration."""
    ByteImages: ImageMode
    FloatImages: ImageMode
    def __str__(self) -> str: ...

class AllocTest:
    """Allocation test result enumeration."""
    Ok: AllocTest
    ImageExceedsLinearTextureLimit: AllocTest
    ImageExceedsLayeredSurfaceLimit: AllocTest
    def __str__(self) -> str: ...

# Main classes
class Config:
    """
    SIFT feature extraction configuration.
    
    This class controls all parameters for SIFT feature detection and description.
    It allows fine-tuning of the algorithm for different types of images and
    performance requirements.
    """
    
    # Properties
    octaves: int
    levels: int
    sigma: float
    verbose: bool
    
    @overload
    def __init__(self) -> None:
        """
        Create a new SIFT configuration with default parameters.
        
        Returns:
            A Config object with the following defaults:
            - octaves: -1 (auto-detect)
            - levels: 3
            - sigma: 1.6
            - threshold: 0.04/3.0
        """
        ...
    
    @overload
    def __init__(self, octaves: int = -1, levels: int = 3, sigma: float = 1.6) -> None:
        """
        Create a new SIFT configuration with specified parameters.
        
        Args:
            octaves: Number of octaves (-1 for auto-detection based on image size)
            levels: Number of levels per octave (typically 3)
            sigma: Initial smoothing sigma value (typically 1.6)
            
        Raises:
            ParameterRangeError: If any parameter is outside valid range
        """
        ...
    
    # Gaussian mode methods
    @overload
    def set_gauss_mode(self, mode: Literal['vlfeat', 'vlfeat-hw-interpolated', 'vlfeat-direct', 'opencv', 'fixed9', 'fixed15']) -> None:
        """
        Set Gaussian filtering mode from string.
        
        Args:
            mode: Gaussian mode string. Valid options:
                - "vlfeat" (default): VLFeat-style computation
                - "vlfeat-hw-interpolated": Hardware-interpolated VLFeat
                - "vlfeat-direct": Direct VLFeat computation  
                - "opencv": OpenCV-style computation
                - "fixed9": Fixed 9-tap filter
                - "fixed15": Fixed 15-tap filter
                
        Raises:
            InvalidEnumError: If mode string is not recognized
        """
        ...
    
    @overload
    def set_gauss_mode(self, mode: GaussMode) -> None:
        """Set Gaussian mode using enum value."""
        ...
    
    def get_gauss_mode(self) -> GaussMode:
        """Get current Gaussian filtering mode."""
        ...
    
    @staticmethod
    def get_gauss_mode_default() -> GaussMode:
        """Get default Gaussian mode."""
        ...
    
    @staticmethod
    def get_gauss_mode_usage() -> str:
        """Get help text for all Gaussian mode options."""
        ...
    
    # Parameter setters with validation
    def set_sigma(self, sigma: float) -> None:
        """
        Set sigma value for Gaussian smoothing.
        
        Args:
            sigma: Sigma value (must be > 0.0 and <= 10.0)
            
        Raises:
            ParameterRangeError: If sigma is outside valid range
        """
        ...
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set feature detection threshold.
        
        Args:
            threshold: Detection threshold (must be >= 0.0)
            
        Raises:
            ParameterRangeError: If threshold is outside valid range
        """
        ...
    
    def set_octaves(self, octaves: int) -> None:
        """Set number of octaves."""
        ...
    
    def set_levels(self, levels: int) -> None:
        """Set number of levels per octave."""
        ...
    
    # Other configuration methods
    def set_sift_mode(self, mode: SiftMode) -> None: ...
    def get_sift_mode(self) -> SiftMode: ...
    def set_log_mode(self, mode: LogMode = ...) -> None: ...
    def get_log_mode(self) -> LogMode: ...
    def set_scaling_mode(self, mode: ScalingMode = ...) -> None: ...
    def get_scaling_mode(self) -> ScalingMode: ...
    
    # String representation
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: Config) -> bool: ...
    def __ne__(self, other: Config) -> bool: ...

class Descriptor:
    """SIFT descriptor with 128-dimensional feature vector."""
    
    def __init__(self) -> None:
        """Create an empty SIFT descriptor with 128 features."""
        ...
    
    @property
    def features(self) -> typing.Any:  # numpy array-like
        """Get the 128-dimensional SIFT feature vector as a NumPy array."""
        ...
    
    def __len__(self) -> Literal[128]:
        """Return the number of features (always 128)."""
        ...
    
    def __getitem__(self, index: int) -> float:
        """Get a feature value by index."""
        ...
    
    def __setitem__(self, index: int, value: float) -> None:
        """Set a feature value by index."""
        ...
    
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Feature:
    """SIFT feature point with location, scale, orientation, and descriptors."""
    
    def __init__(self) -> None:
        """Create an empty SIFT feature."""
        ...
    
    # Read-only properties
    @property
    def debug_octave(self) -> int:
        """Debug octave information."""
        ...
    
    @property
    def xpos(self) -> float:
        """X-coordinate of the feature (scale-adapted)."""
        ...
    
    @property
    def ypos(self) -> float:
        """Y-coordinate of the feature (scale-adapted)."""
        ...
    
    @property
    def sigma(self) -> float:
        """Scale (sigma) of the feature."""
        ...
    
    @property
    def num_ori(self) -> int:
        """Number of orientations for this feature."""
        ...
    
    @property
    def orientation(self) -> typing.Any:  # numpy array-like
        """Get all orientations as a NumPy array."""
        ...
    
    def get_descriptor(self, index: int) -> Descriptor:
        """Get descriptor by index."""
        ...
    
    def get_orientation(self, index: int) -> float:
        """Get orientation by index."""
        ...
    
    def __len__(self) -> int:
        """Return the number of orientations/descriptors."""
        ...
    
    def __getitem__(self, index: int) -> Descriptor:
        """Get descriptor by index (array-style access)."""
        ...
    
    def __iter__(self) -> Iterator[Descriptor]:
        """Iterate over valid descriptors."""
        ...
    
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class FeaturesHost:
    """Container for SIFT features and descriptors."""
    
    @overload
    def __init__(self) -> None:
        """Create an empty features container."""
        ...
    
    @overload  
    def __init__(self, feature_count: int, descriptor_count: int) -> None:
        """Create a features container with specified feature and descriptor counts."""
        ...
    
    def size(self) -> int:
        """Get the number of features."""
        ...
    
    def get_feature_count(self) -> int:
        """Get the number of features."""
        ...
    
    def get_descriptor_count(self) -> int:
        """Get the number of descriptors."""
        ...
    
    def reset(self, feature_count: int, descriptor_count: int) -> None:
        """Reset the container with new feature and descriptor counts."""
        ...
    
    def pin(self) -> None:
        """Pin memory for CUDA operations."""
        ...
    
    def unpin(self) -> None:
        """Unpin memory after CUDA operations."""
        ...
    
    def __len__(self) -> int:
        """Get the number of features."""
        ...
    
    def __getitem__(self, index: int) -> Feature:
        """Get feature by index."""
        ...
    
    def __iter__(self) -> Iterator[Feature]:
        """Iterate over features."""
        ...
    
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# Type alias
Features = FeaturesHost 

class FeaturesDev:
    """
    Container for SIFT features and descriptors in GPU device memory.
    
    FeaturesDev holds SIFT features and descriptors in CUDA device memory,
    optimized for GPU-to-GPU matching operations. It provides zero-copy CuPy array
    views of the device memory for easy integration with GPU computing workflows.
    """
    @overload
    def __init__(self) -> None:
        """Create an empty features container in device memory."""
        ...
    @overload
    def __init__(self, feature_count: int, descriptor_count: int) -> None:
        """Create a features container with specified feature and descriptor counts in device memory."""
        ...
    def size(self) -> int:
        """Get the number of features."""
        ...
    def get_feature_count(self) -> int:
        """Get the number of features."""
        ...
    def get_descriptor_count(self) -> int:
        """Get the number of descriptors."""
        ...
    def reset(self, feature_count: int, descriptor_count: int) -> None:
        """Reset the container with new feature and descriptor counts."""
        ...
    def match(self, other: 'FeaturesDev') -> None:
        """Match features against another FeaturesDev object."""
        ...
    def get_features_array(self) -> 'cupy.ndarray':
        """
        Get features as a zero-copy CuPy array in device memory.
        Returns:
            cupy.ndarray: shape (num_features, 7), dtype float32
        Note:
            This is a view into device memory, valid as long as this object exists.
        """
        ...
    def get_descriptors_array(self) -> 'cupy.ndarray':
        """
        Get descriptors as a zero-copy CuPy array in device memory.
        Returns:
            cupy.ndarray: shape (num_descriptors, 128), dtype float32
        Note:
            This is a view into device memory, valid as long as this object exists.
        """
        ...
    def get_reverse_map_array(self) -> 'cupy.ndarray':
        """
        Get reverse mapping as a zero-copy CuPy array in device memory.
        Returns:
            cupy.ndarray: shape (num_descriptors,), dtype int32
        Note:
            This is a view into device memory, valid as long as this object exists.
        """
        ...
    def to_host(self) -> 'FeaturesHost':
        """Convert device features to host features (device-to-host copy)."""
        ...
    def __len__(self) -> int:
        """Get the number of features."""
        ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ... 

class SiftJob:
    """
    A job for processing an image through the PopSift pipeline.
    
    SiftJob represents an asynchronous task for extracting SIFT features from an image.
    It provides a future-like interface to retrieve the results once processing is complete.
    """
    
    def get(self) -> FeaturesHost:
        """
        Wait for job completion and return the extracted features.
        
        Returns:
            FeaturesHost: The extracted SIFT features
            
        Raises:
            RuntimeError: If the job failed to process
        """
        ...
    
    def get_host(self) -> FeaturesHost:
        """Get features as host memory (alias for get())."""
        ...
    
    def get_dev(self) -> FeaturesDev:
        """
        Get features as device memory (for matching mode).
        
        Returns:
            FeaturesDev: The extracted SIFT features in device memory
            
        Raises:
            RuntimeError: If the job failed to process or is not in matching mode
        """
        ...
    
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class PopSift:
    """
    Main PopSift pipeline for SIFT feature extraction.
    
    PopSift provides a high-performance, GPU-accelerated SIFT feature extraction
    pipeline. It supports both byte (0-255) and float (0-1) image formats and
    can operate in extracting mode (downloads features to host) or matching mode
    (keeps features in device memory for fast matching).
    """
    
    @overload
    def __init__(self, image_mode: ImageMode = ..., device: int = ...) -> None:
        """
        Create a PopSift pipeline with default configuration.
        
        Args:
            image_mode: Type of images to process (ByteImages or FloatImages)
            device: CUDA device ID to use (default: 0)
        """
        ...
    
    @overload
    def __init__(self, config: Config, mode: ProcessingMode = ..., image_mode: ImageMode = ..., device: int = ...) -> None:
        """
        Create a PopSift pipeline with custom configuration.
        
        Args:
            config: SIFT configuration parameters
            mode: Processing mode (ExtractingMode or MatchingMode)
            image_mode: Type of images to process
            device: CUDA device ID to use
        """
        ...
    
    def configure(self, config: Config, force: bool = ...) -> bool:
        """
        Configure the pipeline with new parameters.
        
        Args:
            config: New SIFT configuration
            force: Force reconfiguration even if parameters haven't changed
            
        Returns:
            bool: True if configuration was applied successfully
        """
        ...
    
    def uninit(self) -> None:
        """
        Release all allocated resources.
        
        This should be called when the pipeline is no longer needed to free
        GPU memory and other resources.
        """
        ...
    
    def test_texture_fit(self, width: int, height: int) -> AllocTest:
        """
        Test if the current CUDA device can support the given image dimensions.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            AllocTest: Result indicating if the image size is supported
        """
        ...
    
    def test_texture_fit_error_string(self, err: AllocTest, width: int, height: int) -> str:
        """
        Get a descriptive error message for texture fit test results.
        
        Args:
            err: AllocTest result from test_texture_fit
            width: Original image width
            height: Original image height
            
        Returns:
            str: Human-readable error message with suggestions
        """
        ...
    
    @overload
    def enqueue(self, width: int, height: int, image_data: typing.Any) -> SiftJob:
        """
        Enqueue a byte image for SIFT feature extraction.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            image_data: 2D numpy array with uint8 values (0-255)
            
        Returns:
            SiftJob: Job object for tracking the processing task
            
        Raises:
            ImageError: If image mode is not ByteImages
            RuntimeError: If image dimensions exceed GPU limits
        """
        ...
    
    @overload
    def enqueue(self, width: int, height: int, image_data: typing.Any) -> SiftJob:
        """
        Enqueue a float image for SIFT feature extraction.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            image_data: 2D numpy array with float values (0.0-1.0)
            
        Returns:
            SiftJob: Job object for tracking the processing task
            
        Raises:
            ImageError: If image mode is not FloatImages
            RuntimeError: If image dimensions exceed GPU limits
        """
        ...
    
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ... 