"""
Type stubs for PyPopSift.

This module provides comprehensive type annotations for PyPopSift's enhanced
SIFT feature extraction interface with integrated ImageMode, input validation,
and symmetric CPU/GPU feature support.
"""

import numpy
import numpy.typing as npt
import typing
from typing import overload, Optional, Union

# Type aliases for clarity
UInt8Array = numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]
Float32Array = numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]
Image2D = Union[UInt8Array, Float32Array]

class ImageMode:
    """Image mode enumeration for PopSift processing."""
    ByteImages: ImageMode
    FloatImages: ImageMode
    def __str__(self) -> str: ...

class AllocTest:
    """Allocation test results for GPU texture limits."""
    Ok: AllocTest
    ImageExceedsLinearTextureLimit: AllocTest
    ImageExceedsLayeredSurfaceLimit: AllocTest
    def __str__(self) -> str: ...

class GaussMode:
    """Gaussian filtering modes."""
    VLFeat_Compute: GaussMode
    VLFeat_Relative: GaussMode
    VLFeat_Relative_All: GaussMode
    OpenCV_Compute: GaussMode
    Fixed9: GaussMode
    Fixed15: GaussMode
    def __str__(self) -> str: ...

class SiftMode:
    """SIFT algorithm variants."""
    PopSift: SiftMode
    OpenCV: SiftMode
    VLFeat: SiftMode
    def __str__(self) -> str: ...

class LogMode:
    """Logging modes."""
    # None: LogMode  # Commented out due to conflict with builtin None
    All: LogMode
    def __str__(self) -> str: ...

class ScalingMode:
    """Image scaling modes."""
    ScaleDirect: ScalingMode
    ScaleDefault: ScalingMode
    def __str__(self) -> str: ...

class DescMode:
    """Descriptor computation modes."""
    Loop: DescMode
    ILoop: DescMode
    Grid: DescMode
    IGrid: DescMode
    NoTile: DescMode
    def __str__(self) -> str: ...

class NormMode:
    """Normalization modes."""
    RootSift: NormMode
    Classic: NormMode
    def __str__(self) -> str: ...

class GridFilterMode:
    """Grid filtering modes."""
    RandomScale: GridFilterMode
    LargestScaleFirst: GridFilterMode
    SmallestScaleFirst: GridFilterMode
    def __str__(self) -> str: ...

# Constants
ORIENTATION_MAX_COUNT: int

# Exception classes
class ConfigError(ValueError):
    """Raised when SIFT configuration parameters are invalid."""
    ...

class InvalidEnumError(ConfigError):
    """Raised when an invalid enumeration value is provided."""
    ...

class ParameterRangeError(ConfigError):
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

class Descriptor:
    """SIFT descriptor with 128-dimensional feature vector."""
    
    def __init__(self) -> None:
        """Create an empty SIFT descriptor."""
        ...
    
    @property
    def features(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """Get the 128-dimensional SIFT feature vector as a NumPy array."""
        ...
    
    def __len__(self) -> int:
        """Return the number of features (always 128)."""
        ...
    
    def __getitem__(self, index: int) -> float:
        """Get a feature value by index."""
        ...
    
    def __setitem__(self, index: int, value: float) -> None:
        """Set a feature value by index."""
        ...

class Feature:
    """SIFT feature point with position, scale, and orientations."""
    
    def __init__(self) -> None:
        """Create an empty SIFT feature."""
        ...
    
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
    def orientation(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
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

class FeaturesHost:
    """Container for SIFT features and descriptors in CPU memory."""
    
    @overload
    def __init__(self) -> None: ...
    
    @overload
    def __init__(self, feature_count: int, descriptor_count: int) -> None: ...
    
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
    
    def get_features_array(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """Get features as a zero-copy NumPy array in host memory."""
        ...
    
    def get_descriptors_array(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """Get descriptors as a zero-copy NumPy array in host memory."""
        ...
    
    def to_gpu(self) -> FeaturesDev:
        """Convert host features to device features."""
        ...
    
    def __len__(self) -> int:
        """Get the number of features."""
        ...
    
    def __getitem__(self, index: int) -> Feature:
        """Get feature by index."""
        ...

class FeaturesDev:
    """Container for SIFT features and descriptors in GPU device memory."""
    
    @overload
    def __init__(self) -> None: ...
    
    @overload
    def __init__(self, feature_count: int, descriptor_count: int) -> None: ...
    
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
    
    def match(self, other: FeaturesDev) -> None:
        """Match features against another FeaturesDev object."""
        ...
    
    def get_features_array(self) -> typing.Any:
        """Get features as a zero-copy CuPy array in device memory."""
        ...
    
    def get_descriptors_array(self) -> typing.Any:
        """Get descriptors as a zero-copy CuPy array in device memory."""
        ...
    
    def get_reverse_map_array(self) -> typing.Any:
        """Get reverse mapping as a zero-copy CuPy array in device memory."""
        ...
    
    def to_host(self) -> FeaturesHost:
        """Convert device features to host features."""
        ...
    
    def __len__(self) -> int:
        """Get the number of features."""
        ...

class Config:
    """
    Enhanced SIFT configuration with integrated ImageMode support.
    
    This class provides comprehensive SIFT parameter configuration with
    ImageMode integration for clean API design and automatic input validation.
    """
    
    def __init__(self, octaves: int = -1, levels: int = 3, sigma: float = 1.6) -> None:
        """
        Create a new SIFT configuration.
        
        Args:
            octaves: Number of octaves (-1 for auto-detection)
            levels: Number of levels per octave
            sigma: Initial smoothing sigma value
        """
        ...
    
    def set_image_mode(self, mode: ImageMode) -> None:
        """
        Set the image mode for input validation.
        
        Args:
            mode: ImageMode.ByteImages for uint8 or ImageMode.FloatImages for float32
        """
        ...
    
    def get_image_mode(self) -> ImageMode:
        """Get the current image mode."""
        ...
    
    # SIFT algorithm parameters
    @overload
    def set_gauss_mode(self, mode: typing.Literal['vlfeat', 'vlfeat-hw-interpolated', 'relative', 'vlfeat-direct', 'opencv', 'fixed9', 'fixed15']) -> None: ...
    
    @overload
    def set_gauss_mode(self, mode: GaussMode) -> None: ...
    
    def get_gauss_mode(self) -> GaussMode: ...
    
    def set_sift_mode(self, mode: SiftMode) -> None: ...
    def get_sift_mode(self) -> SiftMode: ...
    
    def set_log_mode(self, mode: LogMode = ...) -> None: ...
    def get_log_mode(self) -> LogMode: ...
    
    def set_scaling_mode(self, mode: ScalingMode = ...) -> None: ...
    def get_scaling_mode(self) -> ScalingMode: ...
    
    def set_octaves(self, octaves: int) -> None: ...
    def set_levels(self, levels: int) -> None: ...
    def set_sigma(self, sigma: float) -> None: ...
    def set_threshold(self, threshold: float) -> None: ...
    def set_initial_blur(self, blur: float) -> None: ...
    def set_filter_max_extrema(self, extrema: int) -> None: ...
    def set_filter_grid_size(self, size: int) -> None: ...
    def set_normalization_multiplier(self, multiplier: float) -> None: ...
    def set_verbose(self, enabled: bool = True) -> None: ...
    
    # Properties
    @property
    def octaves(self) -> int: ...
    
    @property
    def levels(self) -> int: ...
    
    @property
    def sigma(self) -> float: ...
    
    @property
    def verbose(self) -> bool: ...
    
    # Comparison
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...

class SiftJob:
    """
    A job for processing an image through the PopSift pipeline.
    
    SiftJob represents an asynchronous task for extracting SIFT features from an image.
    Returns unified Features objects in GPU memory for optimal performance.
    """
    
    def get(self) -> Features:
        """
        Wait for job completion and return the extracted features.
        
        Returns:
            Features: Unified Features object in GPU memory
        """
        ...
    
    def get_host(self) -> Optional[FeaturesHost]:
        """Get features as FeaturesHost (legacy method)."""
        ...
    
    def get_dev(self) -> Optional[FeaturesDev]:
        """Get features as FeaturesDev (legacy method)."""
        ...

class PopSift:
    """
    Enhanced PopSift pipeline with input validation and automatic uint8 support.
    
    This class provides a clean, validated interface to PopSift with:
    - Automatic uint8 to float32 conversion when needed
    - Input validation for image types and value ranges  
    - ImageMode integrated into Config for logical API design
    - Comprehensive error messages with clear guidance
    """
    
    @overload
    def __init__(self, config: Optional[Config] = None, image_mode: Optional[ImageMode] = None, device: int = 0) -> None:
        """
        Create a PopSift pipeline with enhanced Config.
        
        Args:
            config: Enhanced Config object with ImageMode
            image_mode: Legacy parameter (ignored if config has ImageMode)
            device: CUDA device ID to use
        """
        ...
    
    @overload
    def __init__(self, config: typing.Any, image_mode: ImageMode, device: int = 0) -> None:
        """
        Create a PopSift pipeline with legacy Config (backward compatibility).
        
        Args:
            config: Legacy _pypopsift_impl.Config object
            image_mode: ImageMode for the legacy config
            device: CUDA device ID to use
        """
        ...
    
    @property
    def config(self) -> Config:
        """Get the PopSift configuration."""
        ...
    
    @property
    def device(self) -> int:
        """Get the CUDA device ID."""
        ...
    
    def enqueue(self, image_data: Image2D) -> SiftJob:
        """
        Enqueue an image for SIFT feature extraction with automatic validation.
        
        Args:
            image_data: 2D numpy array with shape (height, width)
                       - uint8: values 0-255 (requires ImageMode.ByteImages)
                       - float32: values 0.0-1.0 (requires ImageMode.FloatImages)
                       
        Returns:
            SiftJob: Job object for tracking the processing task
            
        Raises:
            ValueError: If image validation fails (wrong mode, invalid range, wrong dimensions)
            TypeError: If image is not a numpy array or unsupported dtype
            
        Note:
            - Image dimensions are automatically inferred from array shape
            - uint8 images are automatically converted to float32 for processing
            - Comprehensive validation ensures type safety and clear error messages
        """
        ...
    
    def configure(self, config: Config, force: bool = False) -> bool:
        """Configure the pipeline with new parameters."""
        ...
    
    def uninit(self) -> None:
        """Release all allocated resources."""
        ...
    
    def test_texture_fit(self, width: int, height: int) -> AllocTest:
        """Test if image dimensions are supported."""
        ...
    
    def test_texture_fit_error_string(self, err: AllocTest, width: int, height: int) -> str:
        """Get error message for texture fit test results."""
        ...

class Features:
    """
    Unified SIFT features interface with symmetric CPU/GPU behavior.
    
    This class provides a consistent interface for SIFT features regardless of
    whether they are stored in CPU or GPU memory, with automatic conversions
    and appropriate array types (NumPy for CPU, CuPy for GPU).
    """
    
    @overload
    def __init__(self, feature_count: int, descriptor_count: int) -> None:
        """Create Features with specified counts (CPU memory)."""
        ...
    
    @overload 
    def __init__(self, host_features: FeaturesHost, dev_features: Optional[FeaturesDev]) -> None:
        """Create Features from existing containers."""
        ...
    
    def is_cpu(self) -> bool:
        """Check if features are in CPU memory."""
        ...
    
    def is_gpu(self) -> bool:
        """Check if features are in GPU memory."""
        ...
    
    def cpu(self) -> Features:
        """Get CPU version of features (transfers from GPU if needed)."""
        ...
    
    def gpu(self) -> Features:
        """Get GPU version of features (transfers from CPU if needed)."""
        ...
    
    # Common interface
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
        """Reset with new feature and descriptor counts."""
        ...
    
    # CPU-only operations
    def pin(self) -> None:
        """Pin memory for CUDA operations (CPU mode only)."""
        ...
    
    def unpin(self) -> None:
        """Unpin memory after CUDA operations (CPU mode only)."""
        ...
    
    # GPU-only operations  
    def match(self, other: Features) -> None:
        """Match features against another Features object (GPU mode only)."""
        ...
    
    def get_reverse_map_array(self) -> typing.Any:
        """Get reverse mapping array (GPU mode only, returns CuPy array)."""
        ...
    
    # Symmetric array access
    def get_features_array(self) -> Union[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], typing.Any]:
        """
        Get features as array.
        
        Returns:
            NumPy array for CPU mode, CuPy array for GPU mode.
            Shape is (num_features, 7) with zero-copy views.
        """
        ...
    
    def get_descriptors_array(self) -> Union[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], typing.Any]:
        """
        Get descriptors as array.
        
        Returns:
            NumPy array for CPU mode, CuPy array for GPU mode.
            Shape is (num_descriptors, 128) with zero-copy views.
        """
        ...
    
    # Container operations
    def __len__(self) -> int:
        """Get the number of features."""
        ...
    
    def __getitem__(self, index: int) -> Feature:
        """Get feature by index."""
        ... 