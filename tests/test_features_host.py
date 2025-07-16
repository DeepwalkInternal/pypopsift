import pytest
import pypopsift

def test_features_host_creation():
    features = pypopsift.FeaturesHost()
    assert features is not None
    assert isinstance(features, pypopsift.FeaturesHost)

def test_features_host_parameterized_creation():
    features = pypopsift.FeaturesHost(10, 20)
    assert features is not None
    assert isinstance(features, pypopsift.FeaturesHost)

def test_features_host_properties():
    features = pypopsift.FeaturesHost(5, 10)
    assert features.size() == 5
    assert features.get_feature_count() == 5
    assert features.get_descriptor_count() == 10

def test_features_host_length():
    features = pypopsift.FeaturesHost(3, 6)
    assert len(features) == 3

def test_features_host_indexing():
    features = pypopsift.FeaturesHost(2, 4)
    feature = features[0]
    assert isinstance(feature, pypopsift.Feature)
    feature = features[1]
    assert isinstance(feature, pypopsift.Feature)

def test_features_host_indexing_bounds():
    features = pypopsift.FeaturesHost(2, 4)
    with pytest.raises(IndexError):
        _ = features[2]
    with pytest.raises(IndexError):
        _ = features[-3]

def test_features_host_iteration():
    features = pypopsift.FeaturesHost(3, 6)
    feature_list = list(features)
    assert len(feature_list) == 3
    for feature in feature_list:
        assert isinstance(feature, pypopsift.Feature)

def test_features_host_methods():
    features = pypopsift.FeaturesHost(2, 4)
    features.reset(5, 10)
    assert features.size() == 5
    assert features.get_descriptor_count() == 10
    features.pin()
    features.unpin()

def test_features_host_string_representation():
    features = pypopsift.FeaturesHost(3, 6)
    repr_str = repr(features)
    str_str = str(features)
    assert "FeaturesHost" in repr_str
    assert "3" in str_str
    assert "6" in str_str 