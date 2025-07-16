# type: ignore  # nanobind overloaded methods not recognized by type checker
import pytest
import pypopsift
import numpy as np


class TestImageMode:
    def test_image_mode_values(self):
        assert pypopsift.ImageMode.ByteImages.value == 0
        assert pypopsift.ImageMode.FloatImages.value == 1
    
    def test_image_mode_string_representation(self):
        assert str(pypopsift.ImageMode.ByteImages) == "ByteImages"
        assert str(pypopsift.ImageMode.FloatImages) == "FloatImages"
    
    def test_image_mode_comparison(self):
        assert pypopsift.ImageMode.ByteImages != pypopsift.ImageMode.FloatImages
        assert pypopsift.ImageMode.ByteImages == pypopsift.ImageMode.ByteImages


class TestAllocTest:
    def test_alloc_test_values(self):
        assert pypopsift.AllocTest.Ok.value == 0
        assert pypopsift.AllocTest.ImageExceedsLinearTextureLimit.value == 1
        assert pypopsift.AllocTest.ImageExceedsLayeredSurfaceLimit.value == 2
    
    def test_alloc_test_string_representation(self):
        assert str(pypopsift.AllocTest.Ok) == "Ok"
        assert str(pypopsift.AllocTest.ImageExceedsLinearTextureLimit) == "ImageExceedsLinearTextureLimit"
        assert str(pypopsift.AllocTest.ImageExceedsLayeredSurfaceLimit) == "ImageExceedsLayeredSurfaceLimit"
    
    def test_alloc_test_comparison(self):
        assert pypopsift.AllocTest.Ok != pypopsift.AllocTest.ImageExceedsLinearTextureLimit
        assert pypopsift.AllocTest.Ok == pypopsift.AllocTest.Ok


class TestPopSift:
    def test_popsift_default_constructor(self):
        popsift = pypopsift.PopSift()
        assert isinstance(popsift, pypopsift.PopSift)
        assert str(popsift) == "PopSift"
        assert repr(popsift) == "<PopSift>"
    
    def test_popsift_constructor_with_image_mode(self):
        popsift = pypopsift.PopSift(pypopsift.ImageMode.ByteImages)
        assert isinstance(popsift, pypopsift.PopSift)
        
        popsift = pypopsift.PopSift(pypopsift.ImageMode.FloatImages)
        assert isinstance(popsift, pypopsift.PopSift)
    
    def test_popsift_constructor_with_device(self):
        popsift = pypopsift.PopSift(pypopsift.ImageMode.ByteImages, device=0)
        assert isinstance(popsift, pypopsift.PopSift)
    
    def test_popsift_constructor_with_config(self):
        config = pypopsift.Config(octaves=3, levels=3, sigma=1.6)
        popsift = pypopsift.PopSift(
            config=config,
            mode=pypopsift.ProcessingMode.ExtractingMode,
            image_mode=pypopsift.ImageMode.ByteImages,
            device=0
        )
        assert isinstance(popsift, pypopsift.PopSift)
    
    def test_popsift_constructor_with_config_defaults(self):
        config = pypopsift.Config(octaves=3, levels=3, sigma=1.6)
        popsift = pypopsift.PopSift(config=config)
        assert isinstance(popsift, pypopsift.PopSift)
    
    def test_popsift_configure(self):
        popsift = pypopsift.PopSift()
        config = pypopsift.Config(octaves=4, levels=4, sigma=1.8)
        result = popsift.configure(config)
        assert result is True
    
    def test_popsift_configure_force(self):
        popsift = pypopsift.PopSift()
        config = pypopsift.Config(octaves=4, levels=4, sigma=1.8)
        result = popsift.configure(config, force=True)
        assert result is True
    
    def test_popsift_test_texture_fit(self):
        popsift = pypopsift.PopSift()
        
        result = popsift.test_texture_fit(100, 100)
        assert isinstance(result, pypopsift.AllocTest)
        assert result in [pypopsift.AllocTest.Ok, 
                         pypopsift.AllocTest.ImageExceedsLinearTextureLimit,
                         pypopsift.AllocTest.ImageExceedsLayeredSurfaceLimit]
    
    def test_popsift_test_texture_fit_error_string(self):
        popsift = pypopsift.PopSift()
        
        error_str = popsift.test_texture_fit_error_string(
            pypopsift.AllocTest.Ok, 100, 100
        )
        assert isinstance(error_str, str)
        assert "No error" in error_str
        
        error_str = popsift.test_texture_fit_error_string(
            pypopsift.AllocTest.ImageExceedsLinearTextureLimit, 100, 100
        )
        assert isinstance(error_str, str)
        assert "Cannot load unscaled image" in error_str
        
        error_str = popsift.test_texture_fit_error_string(
            pypopsift.AllocTest.ImageExceedsLayeredSurfaceLimit, 100, 100
        )
        assert isinstance(error_str, str)
        assert "Cannot use" in error_str
    
    def test_popsift_enqueue_byte_image(self):
        popsift = pypopsift.PopSift(pypopsift.ImageMode.ByteImages)
        
        width, height = 64, 64
        image_data = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        
        job = popsift.enqueue(width, height, image_data)
        assert isinstance(job, pypopsift.SiftJob)
    
    def test_popsift_enqueue_float_image(self):
        popsift = pypopsift.PopSift(pypopsift.ImageMode.FloatImages)
        
        width, height = 64, 64
        image_data = np.random.random((height, width)).astype(np.float32)
        
        job = popsift.enqueue(width, height, image_data)
        assert isinstance(job, pypopsift.SiftJob)
    
    def test_popsift_enqueue_invalid_dimensions(self):
        popsift = pypopsift.PopSift(pypopsift.ImageMode.ByteImages)
        
        width, height = 64, 64
        image_data = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Image dimensions don't match"):
            popsift.enqueue(width + 1, height, image_data)
        
        with pytest.raises(ValueError, match="Image dimensions don't match"):
            popsift.enqueue(width, height + 1, image_data)
    
    def test_popsift_enqueue_invalid_array_dimensions(self):
        popsift = pypopsift.PopSift(pypopsift.ImageMode.ByteImages)
        
        image_data = np.random.randint(0, 256, (64,), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Image data must be 2D array"):
            popsift.enqueue(8, 8, image_data)
        
        image_data = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Image data must be 2D array"):
            popsift.enqueue(64, 64, image_data)
    
    def test_popsift_enqueue_wrong_image_mode(self):
        popsift = pypopsift.PopSift(pypopsift.ImageMode.ByteImages)
        
        width, height = 64, 64
        image_data = np.random.random((height, width)).astype(np.float32)
        
        with pytest.raises(pypopsift.ImageError):
            popsift.enqueue(width, height, image_data) 