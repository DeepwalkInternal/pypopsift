import pytest
import numpy as np
import pypopsift

def test_descriptor_creation():
    desc = pypopsift.Descriptor()
    assert desc is not None
    assert isinstance(desc, pypopsift.Descriptor)

def test_descriptor_features_array():
    desc = pypopsift.Descriptor()
    features = desc.features
    assert isinstance(features, np.ndarray)
    assert features.shape == (128,)
    assert features.dtype == np.float32

def test_descriptor_length():
    desc = pypopsift.Descriptor()
    assert len(desc) == 128

def test_descriptor_indexing():
    desc = pypopsift.Descriptor()
    value = desc[0]
    assert isinstance(value, float)
    desc[0] = 1.5
    assert desc[0] == 1.5
    desc[127] = 2.5
    assert desc[127] == 2.5

def test_descriptor_indexing_bounds():
    desc = pypopsift.Descriptor()
    with pytest.raises(IndexError):
        _ = desc[128]
    with pytest.raises(IndexError):
        _ = desc[-129]
    with pytest.raises(IndexError):
        desc[128] = 1.0
    with pytest.raises(IndexError):
        desc[-129] = 1.0

def test_descriptor_string_representation():
    desc = pypopsift.Descriptor()
    desc[0] = 1.0
    desc[1] = 2.0
    desc[127] = 3.0
    repr_str = repr(desc)
    str_str = str(desc)
    assert "Descriptor" in repr_str
    assert "1" in str_str
    assert "2" in str_str
    assert "3" in str_str 