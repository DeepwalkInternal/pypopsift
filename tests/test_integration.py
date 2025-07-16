# type: ignore  # nanobind overloaded methods not recognized by type checker
import pytest
import pypopsift

def test_complete_workflow():
    config = pypopsift.Config(octaves=3, levels=3, sigma=1.6)
    config.set_gauss_mode("vlfeat")
    config.set_sift_mode(pypopsift.SiftMode.PopSift)
    config.set_desc_mode("loop")
    config.set_norm_mode("RootSift")
    config.set_filter_sorting("random")
    config.set_threshold(0.04)
    config.set_edge_limit(10.0)
    config.set_filter_max_extrema(500)
    config.set_filter_grid_size(5)
    assert config.octaves == 3
    features = pypopsift.FeaturesHost(10, 20)
    assert features.size() == 10
    if features.size() > 0:
        feature = features[0]
        assert isinstance(feature, pypopsift.Feature)
        assert hasattr(feature, 'xpos')
        assert hasattr(feature, 'ypos')
        assert hasattr(feature, 'sigma')
        if feature.num_ori > 0:
            desc = feature[0]
            assert isinstance(desc, pypopsift.Descriptor)
            assert len(desc) == 128

def test_large_scale_operations():
    config = pypopsift.Config(octaves=5, levels=4, sigma=1.8)
    config.set_filter_max_extrema(1000)
    config.set_filter_grid_size(10)
    features = pypopsift.FeaturesHost(100, 200)
    assert features.size() == 100
    assert features.get_descriptor_count() == 200
    feature_count = 0
    for feature in features:
        feature_count += 1
        assert isinstance(feature, pypopsift.Feature)
    assert feature_count == 100 