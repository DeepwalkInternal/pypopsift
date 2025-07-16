import pypopsift

def test_orientation_max_count():
    """Test ORIENTATION_MAX_COUNT constant from sift_constants.h."""
    assert hasattr(pypopsift, 'ORIENTATION_MAX_COUNT')
    assert isinstance(pypopsift.ORIENTATION_MAX_COUNT, int)
    assert pypopsift.ORIENTATION_MAX_COUNT == 4  # From sift_constants.h 