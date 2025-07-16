# type: ignore  # nanobind overloaded methods not recognized by type checker
import pytest
import pypopsift
import numpy as np

def test_config_creation():
    config = pypopsift.Config()
    assert config is not None
    assert isinstance(config, pypopsift.Config)

def test_config_default_values():
    config = pypopsift.Config()
    assert config.octaves == -1
    assert config.levels == 3
    assert abs(config.sigma - 1.6) < 1e-6
    assert config.verbose is False

def test_config_parameterized_constructor():
    config = pypopsift.Config(octaves=4, levels=5, sigma=2.0)
    assert config.octaves == 4
    assert config.levels == 5
    assert abs(config.sigma - 2.0) < 1e-6

def test_gauss_mode_string_setting():
    config = pypopsift.Config()
    valid_modes = ["vlfeat", "vlfeat-hw-interpolated", "relative", "vlfeat-direct", "opencv", "fixed9", "fixed15"]
    for mode in valid_modes:
        config.set_gauss_mode(mode)  # type: ignore

def test_gauss_mode_string_setting_invalid():
    config = pypopsift.Config()
    with pytest.raises(pypopsift.InvalidEnumError):
        config.set_gauss_mode("invalid_mode")  # type: ignore

def test_gauss_mode_enum_setting():
    config = pypopsift.Config()
    config.set_gauss_mode(pypopsift.GaussMode.VLFeat_Compute)
    config.set_gauss_mode(pypopsift.GaussMode.OpenCV_Compute)
    config.set_gauss_mode(pypopsift.GaussMode.Fixed9)

def test_desc_mode_string_setting():
    config = pypopsift.Config()
    valid_modes = ["loop", "iloop", "grid", "igrid", "notile"]
    for mode in valid_modes:
        config.set_desc_mode(mode)

def test_desc_mode_string_setting_invalid():
    config = pypopsift.Config()
    with pytest.raises(pypopsift.InvalidEnumError):
        config.set_desc_mode("invalid_mode")

def test_norm_mode_string_setting():
    config = pypopsift.Config()
    config.set_norm_mode("RootSift")
    config.set_norm_mode("classic")

def test_norm_mode_string_setting_invalid():
    config = pypopsift.Config()
    with pytest.raises(pypopsift.InvalidEnumError):
        config.set_norm_mode("invalid_mode")

def test_filter_sorting_string_setting():
    config = pypopsift.Config()
    config.set_filter_sorting("up")
    config.set_filter_sorting("down")
    config.set_filter_sorting("random")

def test_filter_sorting_string_setting_invalid():
    config = pypopsift.Config()
    with pytest.raises(pypopsift.InvalidEnumError):
        config.set_filter_sorting("invalid_mode")

def test_parameter_range_validation_octaves():
    config = pypopsift.Config()
    config.set_octaves(-1)
    config.set_octaves(0)
    config.set_octaves(10)
    with pytest.raises(pypopsift.ParameterRangeError):
        config.set_octaves(-2)

def test_parameter_range_validation_levels():
    config = pypopsift.Config()
    config.set_levels(1)
    config.set_levels(5)
    config.set_levels(10)
    with pytest.raises(pypopsift.ParameterRangeError):
        config.set_levels(0)
    with pytest.raises(pypopsift.ParameterRangeError):
        config.set_levels(11)

def test_parameter_range_validation_sigma():
    config = pypopsift.Config()
    config.set_sigma(0.1)
    config.set_sigma(1.6)
    config.set_sigma(10.0)
    with pytest.raises(pypopsift.ParameterRangeError):
        config.set_sigma(0.0)
    with pytest.raises(pypopsift.ParameterRangeError):
        config.set_sigma(-1.0)
    with pytest.raises(pypopsift.ParameterRangeError):
        config.set_sigma(10.1)

def test_parameter_range_validation_edge_limit():
    config = pypopsift.Config()
    config.set_edge_limit(0.0)
    config.set_edge_limit(10.0)
    config.set_edge_limit(16.0)
    with pytest.raises(pypopsift.ParameterRangeError):
        config.set_edge_limit(-0.1)

def test_parameter_range_validation_threshold():
    config = pypopsift.Config()
    config.set_threshold(0.0)
    config.set_threshold(0.04)
    config.set_threshold(1.0)
    with pytest.raises(pypopsift.ParameterRangeError):
        config.set_threshold(-0.1)

def test_parameter_range_validation_filter_max_extrema():
    config = pypopsift.Config()
    config.set_filter_max_extrema(-1)
    config.set_filter_max_extrema(0)
    config.set_filter_max_extrema(1000)
    with pytest.raises(pypopsift.ParameterRangeError):
        config.set_filter_max_extrema(-2)

def test_parameter_range_validation_filter_grid_size():
    config = pypopsift.Config()
    config.set_filter_grid_size(1)
    config.set_filter_grid_size(5)
    config.set_filter_grid_size(10)
    with pytest.raises(pypopsift.ParameterRangeError):
        config.set_filter_grid_size(0)
    with pytest.raises(pypopsift.ParameterRangeError):
        config.set_filter_grid_size(11)

def test_valid_parameter_setting():
    config = pypopsift.Config()
    config.set_sigma(1.8)
    assert abs(config.sigma - 1.8) < 1e-6
    config.set_threshold(0.01)
    config.set_octaves(5)
    assert config.octaves == 5
    config.set_levels(4)
    assert config.levels == 4
    config.set_edge_limit(10.0)
    config.set_filter_max_extrema(500)
    config.set_filter_grid_size(5)
    config.set_normalization_multiplier(256)
    config.set_downsampling(1.0)
    config.set_initial_blur(0.5)

def test_enum_setting_methods():
    config = pypopsift.Config()
    config.set_sift_mode(pypopsift.SiftMode.PopSift)
    config.set_log_mode(pypopsift.LogMode.All)
    config.set_scaling_mode(pypopsift.ScalingMode.ScaleDefault)
    config.set_desc_mode(pypopsift.DescMode.Loop)
    config.set_norm_mode(pypopsift.NormMode.Classic)
    config.set_filter_sorting(pypopsift.GridFilterMode.RandomScale)

def test_getter_methods():
    config = pypopsift.Config()
    config.get_gauss_mode()
    config.get_sift_mode()
    config.get_log_mode()
    config.get_scaling_mode()
    config.get_desc_mode()
    config.get_norm_mode()
    config.get_filter_sorting()
    config.get_upscale_factor()
    config.get_max_extrema()
    config.get_filter_max_extrema()
    config.get_filter_grid_size()
    config.get_normalization_multiplier()
    config.get_peak_threshold()
    config.get_can_filter_extrema()
    config.has_initial_blur()
    config.get_initial_blur()
    config.if_print_gauss_tables()

def test_static_methods():
    pypopsift.Config.get_gauss_mode_default()
    pypopsift.Config.get_gauss_mode_usage()
    pypopsift.Config.get_norm_mode_default()
    pypopsift.Config.get_norm_mode_usage()

def test_config_comparison():
    config1 = pypopsift.Config(octaves=4, levels=3, sigma=1.6)
    config2 = pypopsift.Config(octaves=4, levels=3, sigma=1.6)
    config3 = pypopsift.Config(octaves=5, levels=3, sigma=1.6)
    assert config1 == config2
    assert config1 != config3

def test_config_string_representation():
    config = pypopsift.Config(octaves=4, levels=3, sigma=1.8)
    repr_str = repr(config)
    str_str = str(config)
    assert "Config" in repr_str
    assert "octaves=4" in repr_str
    assert "levels=3" in repr_str
    assert "sigma=1.8" in repr_str
    assert "SIFT Configuration" in str_str
    assert "Octaves: 4" in str_str
    assert "Levels: 3" in str_str
    assert "Sigma: 1.8" in str_str 