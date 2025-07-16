import pytest
import numpy as np
import pypopsift

def test_feature_creation():
    feature = pypopsift.Feature()
    assert feature is not None
    assert isinstance(feature, pypopsift.Feature)

def test_feature_properties():
    feature = pypopsift.Feature()
    assert hasattr(feature, 'debug_octave')
    assert hasattr(feature, 'xpos')
    assert hasattr(feature, 'ypos')
    assert hasattr(feature, 'sigma')
    assert hasattr(feature, 'num_ori')
    orientations = feature.orientation
    assert isinstance(orientations, np.ndarray)
    assert orientations.shape == (pypopsift.ORIENTATION_MAX_COUNT,)
    assert orientations.dtype == np.float32

def test_feature_descriptor_access():
    feature = pypopsift.Feature()
    if feature.num_ori > 0:
        desc = feature.get_descriptor(0)
        assert isinstance(desc, pypopsift.Descriptor)

def test_feature_orientation_access():
    feature = pypopsift.Feature()
    if feature.num_ori > 0:
        orientation = feature.get_orientation(0)
        assert isinstance(orientation, float)

def test_feature_indexing_bounds():
    feature = pypopsift.Feature()
    with pytest.raises(IndexError):
        feature.get_descriptor(feature.num_ori)
    with pytest.raises(IndexError):
        feature.get_orientation(feature.num_ori)
    with pytest.raises(IndexError):
        feature.get_descriptor(-1)
    with pytest.raises(IndexError):
        feature.get_orientation(-1)

def test_feature_length():
    feature = pypopsift.Feature()
    assert len(feature) == feature.num_ori

def test_feature_array_access():
    feature = pypopsift.Feature()
    if feature.num_ori > 0:
        desc = feature[0]
        assert isinstance(desc, pypopsift.Descriptor)

def test_feature_iteration():
    feature = pypopsift.Feature()
    descriptors = list(feature)
    assert isinstance(descriptors, list)
    assert len(descriptors) <= feature.num_ori

def test_feature_string_representation():
    feature = pypopsift.Feature()
    repr_str = repr(feature)
    str_str = str(feature)
    assert "Feature" in str_str
    assert "x=" in str_str
    assert "y=" in str_str
    assert "sigma=" in str_str 