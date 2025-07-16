import pytest
import pypopsift
import numpy as np
import cupy as cp


def test_features_dev_creation():
    features = pypopsift.FeaturesDev()
    assert features is not None
    assert isinstance(features, pypopsift.FeaturesDev)
    assert features.size() == 0
    assert features.get_feature_count() == 0
    assert features.get_descriptor_count() == 0


def test_features_dev_with_dimensions():
    features = pypopsift.FeaturesDev(5, 10)
    assert features.size() == 5
    assert features.get_feature_count() == 5
    assert features.get_descriptor_count() == 10


def test_features_dev_properties():
    features = pypopsift.FeaturesDev(3, 7)
    assert features.size() == 3
    assert features.get_feature_count() == 3
    assert features.get_descriptor_count() == 7


def test_features_dev_methods():
    features = pypopsift.FeaturesDev(2, 4)
    features.reset(5, 10)
    assert features.size() == 5
    assert features.get_descriptor_count() == 10


def test_features_dev_string_representations():
    features = pypopsift.FeaturesDev(3, 6)
    repr_str = repr(features)
    str_str = str(features)
    
    assert "FeaturesDev" in repr_str
    assert "features=3" in str_str
    assert "descriptors=6" in str_str
    assert "device_memory=True" in str_str


def test_features_dev_length():
    features = pypopsift.FeaturesDev(4, 8)
    assert len(features) == 4


def test_features_dev_cupy_arrays():
    features = pypopsift.FeaturesDev(3, 6)
    
    features_array = features.get_features_array()
    assert isinstance(features_array, cp.ndarray)
    assert features_array.shape == (3, 7)
    assert features_array.dtype == cp.float32
    
    descriptors_array = features.get_descriptors_array()
    assert isinstance(descriptors_array, cp.ndarray)
    assert descriptors_array.shape == (6, 128)
    assert descriptors_array.dtype == cp.float32
    
    reverse_map = features.get_reverse_map_array()
    assert isinstance(reverse_map, cp.ndarray)
    assert reverse_map.shape == (6,)
    assert reverse_map.dtype == cp.int32


def test_features_dev_empty_cupy_arrays():
    features = pypopsift.FeaturesDev(0, 0)
    
    features_array = features.get_features_array()
    assert isinstance(features_array, cp.ndarray)
    assert features_array.shape == (0, 7)
    assert features_array.dtype == cp.float32
    
    descriptors_array = features.get_descriptors_array()
    assert isinstance(descriptors_array, cp.ndarray)
    assert descriptors_array.shape == (0, 128)
    assert descriptors_array.dtype == cp.float32
    
    reverse_map = features.get_reverse_map_array()
    assert isinstance(reverse_map, cp.ndarray)
    assert reverse_map.shape == (0,)
    assert reverse_map.dtype == cp.int32


def test_features_dev_to_host():
    features_dev = pypopsift.FeaturesDev(3, 6)
    host_features = features_dev.to_host()
    
    assert isinstance(host_features, pypopsift.FeaturesHost)
    assert host_features.get_feature_count() == 3
    assert host_features.get_descriptor_count() == 6


def test_features_dev_match():
    features_dev1 = pypopsift.FeaturesDev(2, 4)
    features_dev2 = pypopsift.FeaturesDev(3, 6)
    
    features_dev1.match(features_dev2)


def test_features_dev_integration_workflow():
    features_dev1 = pypopsift.FeaturesDev(2, 4)
    features_dev2 = pypopsift.FeaturesDev(3, 6)
    
    assert features_dev1.get_feature_count() == 2
    assert features_dev1.get_descriptor_count() == 4
    assert features_dev2.get_feature_count() == 3
    assert features_dev2.get_descriptor_count() == 6
    
    features_array1 = features_dev1.get_features_array()
    descriptors_array1 = features_dev1.get_descriptors_array()
    reverse_map1 = features_dev1.get_reverse_map_array()
    
    assert features_array1.shape == (2, 7)
    assert descriptors_array1.shape == (4, 128)
    assert reverse_map1.shape == (4,)
    
    features_dev1.match(features_dev2)
    
    host_features1 = features_dev1.to_host()
    assert isinstance(host_features1, pypopsift.FeaturesHost)
    assert host_features1.get_feature_count() == 2
    assert host_features1.get_descriptor_count() == 4


def test_features_dev_with_popsift_pipeline():
    config = pypopsift.Config()
    popsift = pypopsift.PopSift(config, pypopsift.ProcessingMode.MatchingMode)
    
    width, height = 64, 64
    image_data = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    
    job = popsift.enqueue(width, height, image_data)
    features_dev = job.get_dev()
    
    assert isinstance(features_dev, pypopsift.FeaturesDev)
    assert features_dev.get_feature_count() >= 0
    assert features_dev.get_descriptor_count() >= 0
    
    features_array = features_dev.get_features_array()
    descriptors_array = features_dev.get_descriptors_array()
    reverse_map = features_dev.get_reverse_map_array()
    
    assert isinstance(features_array, cp.ndarray)
    assert isinstance(descriptors_array, cp.ndarray)
    assert isinstance(reverse_map, cp.ndarray)
    
    if features_dev.get_feature_count() > 0:
        assert features_array.shape[0] == features_dev.get_feature_count()
    if features_dev.get_descriptor_count() > 0:
        assert descriptors_array.shape[0] == features_dev.get_descriptor_count()
        assert reverse_map.shape[0] == features_dev.get_descriptor_count()
    
    host_features = features_dev.to_host()
    assert isinstance(host_features, pypopsift.FeaturesHost)
    assert host_features.get_feature_count() == features_dev.get_feature_count()
    assert host_features.get_descriptor_count() == features_dev.get_descriptor_count()


def test_features_dev_error_handling():
    features = pypopsift.FeaturesDev(2, 4)
    
    with pytest.raises(IndexError):
        features_array = features.get_features_array()
        _ = features_array[100, 0]
    
    with pytest.raises(IndexError):
        descriptors_array = features.get_descriptors_array()
        _ = descriptors_array[100, 0]


def test_features_dev_zero_copy_verification():
    features = pypopsift.FeaturesDev(3, 6)
    
    features_array = features.get_features_array()
    descriptors_array = features.get_descriptors_array()
    reverse_map = features.get_reverse_map_array()
    
    assert features_array.data.ptr != 0
    assert descriptors_array.data.ptr != 0
    assert reverse_map.data.ptr != 0
    
    assert features_array.device.id >= 0
    assert descriptors_array.device.id >= 0
    assert reverse_map.device.id >= 0


def test_features_dev_cupy_operations():
    features = pypopsift.FeaturesDev(3, 6)
    
    features_array = features.get_features_array()
    descriptors_array = features.get_descriptors_array()
    reverse_map = features.get_reverse_map_array()
    
    assert cp.all(cp.isfinite(features_array))
    assert cp.all(cp.isfinite(descriptors_array))
    assert cp.all(cp.isfinite(reverse_map))
    
    assert features_array.nbytes == 3 * 7 * 4
    assert descriptors_array.nbytes == 6 * 128 * 4
    assert reverse_map.nbytes == 6 * 4
    
    features_sum = cp.sum(features_array)
    descriptors_sum = cp.sum(descriptors_array)
    reverse_sum = cp.sum(reverse_map)
    
    assert isinstance(features_sum, cp.ndarray)
    assert isinstance(descriptors_sum, cp.ndarray)
    assert isinstance(reverse_sum, cp.ndarray)


def test_features_dev_memory_sharing():
    features = pypopsift.FeaturesDev(2, 4)
    
    features_array1 = features.get_features_array()
    features_array2 = features.get_features_array()
    descriptors_array1 = features.get_descriptors_array()
    descriptors_array2 = features.get_descriptors_array()
    
    assert features_array1.data.ptr == features_array2.data.ptr
    assert descriptors_array1.data.ptr == descriptors_array2.data.ptr
    
    assert features_array1.device == features_array2.device
    assert descriptors_array1.device == descriptors_array2.device


def test_features_dev_dlpack_integration():
    features = pypopsift.FeaturesDev(2, 4)
    
    features_array = features.get_features_array()
    descriptors_array = features.get_descriptors_array()
    
    try:
        dlpack_features = features_array.__dlpack__()
        dlpack_descriptors = descriptors_array.__dlpack__()
        
        features_from_dlpack = cp.from_dlpack(dlpack_features)
        descriptors_from_dlpack = cp.from_dlpack(dlpack_descriptors)
        
        assert cp.array_equal(features_array, features_from_dlpack)
        assert cp.array_equal(descriptors_array, descriptors_from_dlpack)
        
    except AttributeError:
        pytest.skip("DLPack not available in this CuPy version")


def test_features_dev_custom_kernel():
    features = pypopsift.FeaturesDev(3, 6)
    
    features_array = features.get_features_array()
    descriptors_array = features.get_descriptors_array()
    
    kernel_code = '''
    extern "C" __global__
    void test_kernel(const float* input, float* output, int n) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            output[idx] = input[idx] * 2.0f;
        }
    }
    '''
    
    test_kernel = cp.RawKernel(kernel_code, 'test_kernel')
    
    output = cp.empty_like(features_array)
    
    block_size = 256
    grid_size = (features_array.size + block_size - 1) // block_size
    
    test_kernel((grid_size,), (block_size,), (features_array, output, features_array.size))
    
    expected = features_array * 2.0
    assert cp.allclose(output, expected, rtol=1e-5)


def test_features_dev_pytorch_integration():
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
    
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    
    features = pypopsift.FeaturesDev(2, 4)
    
    features_array = features.get_features_array()
    descriptors_array = features.get_descriptors_array()
    
    torch_features = torch.from_dlpack(features_array.__dlpack__())  # type: ignore
    torch_descriptors = torch.from_dlpack(descriptors_array.__dlpack__())  # type: ignore
    
    assert torch_features.shape == (2, 7)
    assert torch_descriptors.shape == (4, 128)
    assert torch_features.dtype == torch.float32
    assert torch_descriptors.dtype == torch.float32
    
    assert torch_features.device.type == 'cuda'
    assert torch_descriptors.device.type == 'cuda'


def test_features_dev_lifetime_management():
    features = pypopsift.FeaturesDev(2, 4)
    
    features_array = features.get_features_array()
    descriptors_array = features.get_descriptors_array()
    
    features_ptr = features_array.data.ptr
    descriptors_ptr = descriptors_array.data.ptr
    
    features_sum = cp.sum(features_array)
    descriptors_sum = cp.sum(descriptors_array)
    
    assert features_array.data.ptr == features_ptr
    assert descriptors_array.data.ptr == descriptors_ptr
    
    assert cp.all(cp.isfinite(features_array))
    assert cp.all(cp.isfinite(descriptors_array))


def test_features_dev_empty_arrays_zero_copy():
    features = pypopsift.FeaturesDev(0, 0)
    
    features_array = features.get_features_array()
    descriptors_array = features.get_descriptors_array()
    reverse_map = features.get_reverse_map_array()
    
    assert features_array.device.id >= 0
    assert descriptors_array.device.id >= 0
    assert reverse_map.device.id >= 0
    
    assert features_array.size == 0
    assert descriptors_array.size == 0
    assert reverse_map.size == 0


if __name__ == "__main__":
    test_features_dev_creation()
    test_features_dev_with_dimensions()
    test_features_dev_properties()
    test_features_dev_methods()
    test_features_dev_string_representations()
    test_features_dev_length()
    test_features_dev_to_host()
    test_features_dev_match()
    test_features_dev_integration_workflow()
    test_features_dev_error_handling()
    
    test_features_dev_cupy_arrays()
    test_features_dev_empty_cupy_arrays()
    test_features_dev_with_popsift_pipeline()
    test_features_dev_zero_copy_verification()
    test_features_dev_cupy_operations()
    test_features_dev_memory_sharing()
    test_features_dev_dlpack_integration()
    test_features_dev_custom_kernel()
    test_features_dev_pytorch_integration()
    test_features_dev_lifetime_management()
    test_features_dev_empty_arrays_zero_copy()
    
    print("All FeaturesDev tests passed!") 