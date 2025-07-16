"""
Python bindings for the PopSift library, a real-time SIFT implementation in CUDA.

This module provides Python access to the PopSift library for SIFT feature extraction
and matching using CUDA acceleration.
"""

from ._pypopsift_impl import ( # type: ignore
    # Constants
    ORIENTATION_MAX_COUNT,
    
    # Exception classes
    ConfigError,
    InvalidEnumError,
    ParameterRangeError,
    MemoryError,
    CudaError,
    ImageError,
    UnsupportedOperationError,
    LogicError,
    
    # Enum classes
    GaussMode,
    SiftMode,
    LogMode,
    ScalingMode,
    DescMode,
    NormMode,
    GridFilterMode,
    ProcessingMode,
    ImageMode,
    AllocTest,
    
    # Main classes
    Config,
    Descriptor,
    Feature,
    FeaturesHost,
    Features,
    FeaturesDev,
    PopSift,
    SiftJob,
)

__version__ = "0.1.0"
__all__ = [
    # Constants
    "ORIENTATION_MAX_COUNT",
    
    # Exception classes
    "ConfigError",
    "InvalidEnumError", 
    "ParameterRangeError",
    "MemoryError",
    "CudaError",
    "ImageError",
    "UnsupportedOperationError",
    "LogicError",
    
    # Enum classes
    "GaussMode",
    "SiftMode", 
    "LogMode",
    "ScalingMode",
    "DescMode",
    "NormMode",
    "GridFilterMode",
    "ProcessingMode",
    "ImageMode",
    "AllocTest",
    
    # Main classes
    "Config",
    "Descriptor",
    "Feature",
    "FeaturesHost",
    "Features",
    "FeaturesDev",
    "PopSift",
    "SiftJob",
]
