import pypopsift
import pytest

def test_exception_availability():
    exceptions = [
        pypopsift.ConfigError,
        pypopsift.InvalidEnumError,
        pypopsift.ParameterRangeError,
        pypopsift.MemoryError,
        pypopsift.CudaError,
        pypopsift.ImageError,
        pypopsift.UnsupportedOperationError,
        pypopsift.LogicError,
    ]
    for exc in exceptions:
        assert exc is not None
        assert isinstance(exc, type)

def test_exception_inheritance():
    assert issubclass(pypopsift.ConfigError, ValueError)
    assert issubclass(pypopsift.InvalidEnumError, pypopsift.ConfigError)
    assert issubclass(pypopsift.ParameterRangeError, pypopsift.ConfigError)
    assert issubclass(pypopsift.MemoryError, MemoryError)
    assert issubclass(pypopsift.CudaError, RuntimeError)
    assert issubclass(pypopsift.ImageError, RuntimeError)
    assert issubclass(pypopsift.UnsupportedOperationError, NotImplementedError)
    assert issubclass(pypopsift.LogicError, RuntimeError)

def test_exception_names():
    assert pypopsift.ConfigError.__name__ == "ConfigError"
    assert pypopsift.InvalidEnumError.__name__ == "InvalidEnumError"
    assert pypopsift.ParameterRangeError.__name__ == "ParameterRangeError"
    assert pypopsift.MemoryError.__name__ == "MemoryError"
    assert pypopsift.CudaError.__name__ == "CudaError"
    assert pypopsift.ImageError.__name__ == "ImageError"
    assert pypopsift.UnsupportedOperationError.__name__ == "UnsupportedOperationError"
    assert pypopsift.LogicError.__name__ == "LogicError" 