#!/usr/bin/env python3
"""
Test script to verify that the new exception classes work correctly.
"""

import pypopsift
import numpy as np

def test_config_exceptions():
    """Test configuration-related exceptions."""
    print("Testing configuration exceptions...")
    
    config = pypopsift.Config()
    
    # Test ParameterRangeError
    try:
        config.set_sigma(-1.0)  # Invalid sigma value
        assert False, "Should have raised ParameterRangeError"
    except pypopsift.ParameterRangeError as e:
        print(f"✓ Caught ParameterRangeError: {e}")
    
    # Test InvalidEnumError
    try:
        config.set_gauss_mode("InvalidMode")  # Invalid enum value
        assert False, "Should have raised InvalidEnumError"
    except pypopsift.InvalidEnumError as e:
        print(f"✓ Caught InvalidEnumError: {e}")
    
    # Test ParameterRangeError for threshold
    try:
        config.set_threshold(-0.1)  # Invalid threshold
        assert False, "Should have raised ParameterRangeError"
    except pypopsift.ParameterRangeError as e:
        print(f"✓ Caught ParameterRangeError: {e}")

def test_memory_exceptions():
    """Test memory-related exceptions."""
    print("\nTesting memory exceptions...")
    
    # Note: We can't easily trigger memory errors in a test environment
    # but we can verify the exception class exists
    print("✓ MemoryError class is available")
    print(f"  MemoryError: {pypopsift.MemoryError}")

def test_cuda_exceptions():
    """Test CUDA-related exceptions."""
    print("\nTesting CUDA exceptions...")
    
    # Note: We can't easily trigger CUDA errors in a test environment
    # but we can verify the exception class exists
    print("✓ CudaError class is available")
    print(f"  CudaError: {pypopsift.CudaError}")

def test_image_exceptions():
    """Test image-related exceptions."""
    print("\nTesting image exceptions...")
    
    # Note: We can't easily trigger image errors in a test environment
    # but we can verify the exception class exists
    print("✓ ImageError class is available")
    print(f"  ImageError: {pypopsift.ImageError}")

def test_unsupported_operation_exceptions():
    """Test unsupported operation exceptions."""
    print("\nTesting unsupported operation exceptions...")
    
    # Note: We can't easily trigger these in a test environment
    # but we can verify the exception class exists
    print("✓ UnsupportedOperationError class is available")
    print(f"  UnsupportedOperationError: {pypopsift.UnsupportedOperationError}")

def test_logic_exceptions():
    """Test logic-related exceptions."""
    print("\nTesting logic exceptions...")
    
    # Note: We can't easily trigger logic errors in a test environment
    # but we can verify the exception class exists
    print("✓ LogicError class is available")
    print(f"  LogicError: {pypopsift.LogicError}")

def test_exception_hierarchy():
    """Test that exceptions have the correct hierarchy."""
    print("\nTesting exception hierarchy...")
    
    # Test that our exceptions inherit from the correct base classes
    assert issubclass(pypopsift.ConfigError, ValueError), "ConfigError should inherit from ValueError"
    assert issubclass(pypopsift.InvalidEnumError, ValueError), "InvalidEnumError should inherit from ValueError"
    assert issubclass(pypopsift.ParameterRangeError, ValueError), "ParameterRangeError should inherit from ValueError"
    
    assert issubclass(pypopsift.MemoryError, MemoryError), "MemoryError should inherit from MemoryError"
    assert issubclass(pypopsift.CudaError, RuntimeError), "CudaError should inherit from RuntimeError"
    assert issubclass(pypopsift.ImageError, RuntimeError), "ImageError should inherit from RuntimeError"
    assert issubclass(pypopsift.UnsupportedOperationError, NotImplementedError), "UnsupportedOperationError should inherit from NotImplementedError"
    assert issubclass(pypopsift.LogicError, RuntimeError), "LogicError should inherit from RuntimeError"
    
    print("✓ All exception classes have correct inheritance")

if __name__ == "__main__":
    try:
        test_config_exceptions()
        test_memory_exceptions()
        test_cuda_exceptions()
        test_image_exceptions()
        test_unsupported_operation_exceptions()
        test_logic_exceptions()
        test_exception_hierarchy()
        
        print("\n🎉 All exception tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 