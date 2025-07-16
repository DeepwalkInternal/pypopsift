# type: ignore  # nanobind overloaded methods not recognized by type checker
import pytest
import numpy as np
import pypopsift

def test_empty_features_host():
    features = pypopsift.FeaturesHost(0, 0)
    assert features.size() == 0
    assert len(features) == 0
    feature_list = list(features)
    assert len(feature_list) == 0

def test_single_feature():
    features = pypopsift.FeaturesHost(1, 1)
    assert features.size() == 1
    feature = features[0]
    assert isinstance(feature, pypopsift.Feature)

def test_config_extreme_values():
    config = pypopsift.Config()
    config.set_octaves(10)
    config.set_levels(10)
    config.set_sigma(10.0)
    config.set_threshold(1.0)
    config.set_edge_limit(100.0)
    config.set_filter_max_extrema(10000)
    config.set_filter_grid_size(10)
    assert config.octaves == 10
    assert config.levels == 10
    assert config.sigma == 10.0

def test_descriptor_edge_cases():
    desc = pypopsift.Descriptor()
    desc[0] = float('inf')
    desc[1] = float('-inf')
    desc[2] = float('nan')
    assert np.isinf(desc[0])
    assert np.isinf(desc[1])
    assert np.isnan(desc[2])

def test_string_case_sensitivity():
    config = pypopsift.Config()
    with pytest.raises(pypopsift.InvalidEnumError):
        config.set_gauss_mode("VLFEAT")
    with pytest.raises(pypopsift.InvalidEnumError):
        config.set_desc_mode("LOOP")
    with pytest.raises(pypopsift.InvalidEnumError):
        config.set_norm_mode("rootsift")

def test_boundary_values():
    config = pypopsift.Config()
    config.set_octaves(-1)
    config.set_levels(1)
    config.set_levels(10)
    config.set_sigma(0.1)
    config.set_sigma(10.0)
    config.set_threshold(0.0)
    config.set_edge_limit(0.0)
    config.set_filter_max_extrema(-1)
    config.set_filter_grid_size(1)
    config.set_filter_grid_size(10) 