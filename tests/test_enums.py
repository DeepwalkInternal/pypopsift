import pypopsift

def test_gauss_mode_enum():
    assert hasattr(pypopsift, 'GaussMode')
    assert pypopsift.GaussMode.VLFeat_Compute is not None
    assert pypopsift.GaussMode.VLFeat_Relative is not None
    assert pypopsift.GaussMode.VLFeat_Relative_All is not None
    assert pypopsift.GaussMode.OpenCV_Compute is not None
    assert pypopsift.GaussMode.Fixed9 is not None
    assert pypopsift.GaussMode.Fixed15 is not None
    assert str(pypopsift.GaussMode.VLFeat_Compute) == "VLFeat_Compute"
    assert str(pypopsift.GaussMode.OpenCV_Compute) == "OpenCV_Compute"
    assert str(pypopsift.GaussMode.Fixed9) == "Fixed9"

def test_sift_mode_enum():
    assert hasattr(pypopsift, 'SiftMode')
    assert pypopsift.SiftMode.PopSift is not None
    assert pypopsift.SiftMode.OpenCV is not None
    assert pypopsift.SiftMode.VLFeat is not None
    assert str(pypopsift.SiftMode.PopSift) == "PopSift"
    assert str(pypopsift.SiftMode.OpenCV) == "OpenCV"
    assert str(pypopsift.SiftMode.VLFeat) == "VLFeat"

def test_log_mode_enum():
    assert hasattr(pypopsift, 'LogMode')
    assert getattr(pypopsift.LogMode, 'None') is not None
    assert pypopsift.LogMode.All is not None
    assert str(getattr(pypopsift.LogMode, 'None')) == "None"
    assert str(pypopsift.LogMode.All) == "All"

def test_scaling_mode_enum():
    assert hasattr(pypopsift, 'ScalingMode')
    assert pypopsift.ScalingMode.ScaleDirect is not None
    assert pypopsift.ScalingMode.ScaleDefault is not None
    assert str(pypopsift.ScalingMode.ScaleDirect) == "ScaleDirect"
    assert str(pypopsift.ScalingMode.ScaleDefault) == "ScaleDefault"

def test_desc_mode_enum():
    assert hasattr(pypopsift, 'DescMode')
    assert pypopsift.DescMode.Loop is not None
    assert pypopsift.DescMode.ILoop is not None
    assert pypopsift.DescMode.Grid is not None
    assert pypopsift.DescMode.IGrid is not None
    assert pypopsift.DescMode.NoTile is not None
    assert str(pypopsift.DescMode.Loop) == "Loop"
    assert str(pypopsift.DescMode.ILoop) == "ILoop"
    assert str(pypopsift.DescMode.Grid) == "Grid"
    assert str(pypopsift.DescMode.IGrid) == "IGrid"
    assert str(pypopsift.DescMode.NoTile) == "NoTile"

def test_norm_mode_enum():
    assert hasattr(pypopsift, 'NormMode')
    assert pypopsift.NormMode.RootSift is not None
    assert pypopsift.NormMode.Classic is not None
    assert str(pypopsift.NormMode.RootSift) == "RootSift"
    assert str(pypopsift.NormMode.Classic) == "Classic"

def test_grid_filter_mode_enum():
    assert hasattr(pypopsift, 'GridFilterMode')
    assert pypopsift.GridFilterMode.RandomScale is not None
    assert pypopsift.GridFilterMode.LargestScaleFirst is not None
    assert pypopsift.GridFilterMode.SmallestScaleFirst is not None
    assert str(pypopsift.GridFilterMode.RandomScale) == "RandomScale"
    assert str(pypopsift.GridFilterMode.LargestScaleFirst) == "LargestScaleFirst"
    assert str(pypopsift.GridFilterMode.SmallestScaleFirst) == "SmallestScaleFirst"

def test_processing_mode_enum():
    assert hasattr(pypopsift, 'ProcessingMode')
    assert pypopsift.ProcessingMode.ExtractingMode is not None
    assert pypopsift.ProcessingMode.MatchingMode is not None
    assert str(pypopsift.ProcessingMode.ExtractingMode) == "ExtractingMode"
    assert str(pypopsift.ProcessingMode.MatchingMode) == "MatchingMode" 