"""
Type stubs for pypopsift - Python bindings for PopSift SIFT library.

This file provides comprehensive type information for static analysis tools,
IDEs, and type checkers while preserving the rich documentation from the
underlying nanobind C++ implementation.
"""

from typing import Literal, overload
import typing
from collections.abc import Iterator

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