# PyPopSift

Python bindings for the PopSift library, a real-time SIFT implementation in CUDA.

PyPopSift provides high-performance, GPU-accelerated SIFT feature extraction and matching using CUDA. It offers zero-copy CuPy integration for seamless GPU computing workflows.

## Features

- **CUDA-accelerated SIFT**: Real-time feature extraction using GPU
- **Zero-copy CuPy integration**: Direct GPU memory access without copying
- **Multiple image formats**: Support for both byte (0-255) and float (0-1) images
- **Flexible configuration**: Extensive parameter tuning for different use cases
- **Comprehensive API**: Full access to all PopSift functionality

## Requirements

- Python 3.12+
- CUDA 12.x compatible GPU
- Linux (tested on Ubuntu)

## Installation

### For Users

Install the package directly:

```bash
pip install pypopsift
```

### For Developers

1. Clone the repository:
```bash
git clone <repository-url>
cd pypopsift
```

2. Install build dependencies:
```bash
pip install nanobind scikit-build-core[pyproject]
```

3. Install in development mode:
```bash
pip install --no-build-isolation -ve .
```

For automatic rebuilds during development:
```bash
pip install --no-build-isolation -Ceditable.rebuild=true -ve .
```

## Quick Start

### Basic SIFT Feature Extraction

```python
import pypopsift
import numpy as np

# Create a PopSift pipeline
popsift = pypopsift.PopSift()

# Create a test image (grayscale, 0-255)
image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

# Extract SIFT features
job = popsift.enqueue(640, 480, image)
features = job.get()

print(f"Found {len(features)} features")
```

### Using CuPy for Zero-Copy GPU Operations

```python
import pypopsift
import cupy as cp
import numpy as np

# Create PopSift in matching mode for GPU operations
config = pypopsift.Config()
popsift = pypopsift.PopSift(config, mode=pypopsift.ProcessingMode.MatchingMode)

# Create image on GPU
image = cp.random.randint(0, 256, (480, 640), dtype=cp.uint8)

# Extract features directly to GPU memory
job = popsift.enqueue(640, 480, image.get())
features_dev = job.get_dev()

# Get zero-copy CuPy arrays
features_array = features_dev.get_features_array()  # shape: (num_features, 7)
descriptors_array = features_dev.get_descriptors_array()  # shape: (num_descriptors, 128)

# Perform GPU operations without copying
distances = cp.linalg.norm(descriptors_array[None, :, :] - descriptors_array[:, None, :], axis=2)
```

### Custom Configuration

```python
import pypopsift

# Create custom SIFT configuration
config = pypopsift.Config(
    octaves=4,      # Number of octaves
    levels=3,       # Levels per octave
    sigma=1.6       # Initial smoothing
)

# Set additional parameters
config.set_threshold(0.04)  # Feature detection threshold
config.set_gauss_mode("vlfeat")  # Gaussian filtering mode

# Create pipeline with custom config
popsift = pypopsift.PopSift(config)
```

## API Reference

### Main Classes

- **`PopSift`**: Main pipeline for SIFT feature extraction
- **`Config`**: Configuration for SIFT parameters
- **`FeaturesHost`**: Container for features in host memory
- **`FeaturesDev`**: Container for features in GPU device memory
- **`Feature`**: Individual SIFT feature with location, scale, and descriptors
- **`Descriptor`**: 128-dimensional SIFT descriptor vector

### Key Methods

#### PopSift
- `enqueue(width, height, image_data)` → `SiftJob`: Queue image for processing
- `configure(config, force=False)` → `bool`: Update configuration
- `test_texture_fit(width, height)` → `AllocTest`: Check if image size is supported

#### FeaturesDev (GPU Features)
- `get_features_array()` → `cupy.ndarray`: Zero-copy features array
- `get_descriptors_array()` → `cupy.ndarray`: Zero-copy descriptors array
- `get_reverse_map_array()` → `cupy.ndarray`: Zero-copy reverse mapping
- `match(other)` → `None`: Match against another FeaturesDev
- `to_host()` → `FeaturesHost`: Convert to host memory

## Configuration Options

### Gaussian Filtering Modes
- `"vlfeat"`: VLFeat-style computation (default)
- `"vlfeat-hw-interpolated"`: Hardware-interpolated VLFeat
- `"vlfeat-direct"`: Direct VLFeat computation
- `"opencv"`: OpenCV-style computation
- `"fixed9"`: Fixed 9-tap filter
- `"fixed15"`: Fixed 15-tap filter

### Processing Modes
- `ExtractingMode`: Download features to host memory
- `MatchingMode`: Keep features in GPU memory for matching

### Image Modes
- `ByteImages`: Process 0-255 byte images
- `FloatImages`: Process 0.0-1.0 float images

## Examples

See the `tests/` directory for comprehensive examples covering:
- Basic feature extraction
- CuPy integration and zero-copy operations
- Custom kernel usage
- DLPack and PyTorch integration
- Memory management and lifetime handling

## Development

### Building from Source

The package uses scikit-build-core for building. The CMakeLists.txt file is configured to work with the PopSift C++ library included as a submodule.

```bash
# Install build dependencies
pip install nanobind scikit-build-core[pyproject]

# Build and install in development mode
pip install --no-build-isolation -ve .

# For automatic rebuilds during development
pip install --no-build-isolation -Ceditable.rebuild=true -ve .
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=pypopsift --cov-report=html

# Run specific test file
pytest tests/test_features_dev.py -v
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
