# PopSift Python Bindings

Python bindings for the PopSift library, a real-time SIFT (Scale-Invariant Feature Transform) implementation using CUDA for GPU acceleration.

## Features

- **GPU-accelerated SIFT feature extraction** using CUDA
- **Real-time performance** optimized for video processing
- **Unified Features interface** with seamless CPU/GPU memory transfer
- **CuPy integration** for zero-copy GPU array access
- **Comprehensive configuration options** for fine-tuning SIFT parameters
- **Multiple SIFT modes** (PopSift, OpenCV, VLFeat compatibility)

## Installation

```bash
pip install pypopsift
```

## Quick Start

### Basic Usage

```python
import pypopsift
import numpy as np

# Create a PopSift instance
popsift = pypopsift.PopSift()

# Load an image
image_data = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

# Extract SIFT features
job = popsift.enqueue(image_data)  # Dimensions automatically inferred
features = job.get()  # Get features in GPU memory (unified Features class)

# Access feature information
print(f"Found {features.get_feature_count()} features")
print(f"Generated {features.get_descriptor_count()} descriptors")

### Improved API - Automatic Dimension Inference

No need to manually specify image dimensions! PopSift automatically infers width and height from your NumPy array:

```python
import pypopsift
import numpy as np

# Create test image (height=480, width=640)
image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

# Old way (still works for backward compatibility):
# job = popsift.enqueue(640, 480, image)

# New way - cleaner and less error-prone:
popsift = pypopsift.PopSift()
job = popsift.enqueue(image)  # Dimensions automatically inferred!
features = job.get()

# Works with both uint8 (0-255) and float32 (0.0-1.0) images
float_image = np.random.rand(480, 640).astype(np.float32)
job = popsift.enqueue(float_image)  # Just as easy!
```

**Benefits:**
- 🎯 **Cleaner API**: No redundant dimension parameters
- 🛡️ **Error Prevention**: No risk of dimension mismatch
- 🐍 **More Pythonic**: Follows NumPy conventions
- ⚡ **Same Performance**: Zero overhead

### Unified Features Interface

The `Features` class provides a unified interface that can hold features in either CPU or GPU memory, with seamless transfer and **symmetric array access**:

```python
import pypopsift
import numpy as np

# Create features in CPU memory (default)
features = pypopsift.Features(100, 200)
print(features)  # Features(CPU, features=100, descriptors=200)

# Transfer to GPU
features_gpu = features.gpu()
print(features_gpu)  # Features(GPU, features=100, descriptors=200)

# Transfer back to CPU
features_cpu = features_gpu.cpu()
print(features_cpu)  # Features(CPU, features=100, descriptors=200)

# Check memory location
print(features.is_cpu())  # True
print(features_gpu.is_gpu())  # True
```

### Symmetric Array Access

Both CPU and GPU modes support direct array access with appropriate array types:

```python
# CPU mode - returns NumPy arrays
cpu_features = pypopsift.Features(100, 200)
features_array = cpu_features.get_features_array()      # NumPy array (100, 7)
descriptors_array = cpu_features.get_descriptors_array()  # NumPy array (200, 128)

assert isinstance(features_array, np.ndarray)
assert features_array.shape == (100, 7)  # [debug_octave, xpos, ypos, sigma, num_ori, orientation[0], orientation[1]]
assert descriptors_array.shape == (200, 128)  # Standard SIFT descriptors

# GPU mode - returns CuPy arrays
gpu_features = cpu_features.gpu()
gpu_features_array = gpu_features.get_features_array()      # CuPy array (100, 7)
gpu_descriptors_array = gpu_features.get_descriptors_array()  # CuPy array (200, 128)
reverse_map = gpu_features.get_reverse_map_array()          # CuPy array (200,) - GPU only

# Same API, different array types!
assert hasattr(gpu_features_array, '__cuda_array_interface__')  # CuPy array
```

### Real SIFT Extraction Workflow

```python
import pypopsift
import numpy as np
from PIL import Image

# Load and prepare image
with Image.open('image.jpg') as img:
    img_gray = img.convert('L')
    image_data = np.array(img_gray, dtype=np.uint8)

# Extract SIFT features
popsift = pypopsift.PopSift()
job = popsift.enqueue(image_data)  # Dimensions automatically inferred

# Get features in GPU memory (recommended for performance)
dev_features = job.get_dev()
features = pypopsift.Features(dev_features)

# Transfer to CPU for analysis if needed
cpu_features = features.cpu()

# Access individual features
for i in range(min(5, cpu_features.get_feature_count())):
    feature = cpu_features[i]
    print(f"Feature {i}: x={feature.xpos:.2f}, y={feature.ypos:.2f}, sigma={feature.sigma:.2f}")
    
    # Access descriptors
    for j in range(feature.num_ori):
        descriptor = feature[j]
        print(f"  Descriptor {j}: {len(descriptor)} features")
```

### Advanced Array Operations

The symmetric array interface enables powerful array-based processing:

```python
import pypopsift
import numpy as np
import cupy as cp

# Extract features
popsift = pypopsift.PopSift()
job = popsift.enqueue(image_data)  # Dimensions automatically inferred
features = job.get()  # Unified Features object (GPU by default)

# Process with same code, different backends
def process_features(features):
    """Works for both CPU and GPU features."""
    features_array = features.get_features_array()
    descriptors_array = features.get_descriptors_array()
    
    if features.is_cpu():
        # NumPy operations
        mean_descriptor = np.mean(descriptors_array, axis=1)
        feature_scales = features_array[:, 3]  # sigma column
    else:
        # CuPy operations
        mean_descriptor = cp.mean(descriptors_array, axis=1)
        feature_scales = features_array[:, 3]  # sigma column
        
        # GPU-only operations
        reverse_map = features.get_reverse_map_array()
    
    return mean_descriptor, feature_scales

# Process in GPU memory for performance
gpu_mean, gpu_scales = process_features(features)

# Transfer to CPU for NumPy ecosystem processing
cpu_features = features.cpu()
cpu_mean, cpu_scales = process_features(cpu_features)
```

## Configuration

```python
import pypopsift

# Create configuration
config = pypopsift.Config(
    octaves=4,      # Number of octaves (-1 for auto)
    levels=3,       # Levels per octave
    sigma=1.6       # Initial smoothing
)

# Set additional parameters
config.set_threshold(0.04/3.0)
config.set_edge_limit(10.0)
config.set_gauss_mode("vlfeat")
config.set_desc_mode("loop")
config.set_norm_mode("classic")

# Create PopSift with configuration
popsift = pypopsift.PopSift(config)
```

## API Reference

### Main Classes

- **`PopSift`**: Main SIFT extraction pipeline
- **`Features`**: Unified interface for CPU/GPU features
- **`FeaturesHost`**: CPU memory features container
- **`FeaturesDev`**: GPU memory features container
- **`Config`**: SIFT configuration parameters
- **`SiftJob`**: Asynchronous job for feature extraction

### Key Methods

#### Features Class
- `cpu()`: Transfer features to CPU memory
- `gpu()`: Transfer features to GPU memory
- `is_cpu()`: Check if features are in CPU memory
- `is_gpu()`: Check if features are in GPU memory
- `get_feature_count()`: Get number of features
- `get_descriptor_count()`: Get number of descriptors

#### PopSift Class
- `enqueue(width, height, image_data)`: Queue image for processing
- `configure(config, force=False)`: Update configuration
- `test_texture_fit(width, height)`: Test GPU compatibility

#### SiftJob Class
- `get_dev()`: Get features in GPU memory (recommended)
- `get()` or `get_host()`: Get features in CPU memory

## Examples

See the `examples/` directory for complete working examples:

- `unified_features_example.py`: Demonstrates the unified Features interface
- Additional examples coming soon...

## Requirements

- Python 3.8+
- CUDA 11.0+
- Compatible NVIDIA GPU
- CuPy (for GPU array access)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
