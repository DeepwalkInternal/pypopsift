#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>  
#include <nanobind/make_iterator.h>
#include <nanobind/stl/string.h>
#include <nanobind/typing.h>
#include <nanobind/operators.h>
#include <popsift/sift_extremum.h>
#include <popsift/features.h>
#include <popsift/sift_constants.h>
#include <popsift/sift_conf.h>
#include <popsift/popsift.h>
#include <sstream>
#include <cuda_runtime.h>

// Helper to get current CUDA device
int get_cuda_device() {
    int device = 0;
    cudaGetDevice(&device);
    return device;
}

namespace nb = nanobind;

NB_MODULE(_pypopsift_impl, m) {
    m.doc() = R"pbdoc(
        Python bindings for the PopSift library
        
        PopSift is a real-time SIFT (Scale-Invariant Feature Transform) implementation
        using CUDA for GPU acceleration. This module provides Python access to
        PopSift's feature extraction and matching capabilities.
        
        Example:
            >>> import pypopsift
            >>> config = pypopsift.Config()
            >>> config.set_sigma(1.6)
            >>> config.set_octaves(4)
    )pbdoc";

    m.attr("ORIENTATION_MAX_COUNT") = nb::int_(ORIENTATION_MAX_COUNT);
    m.attr("__doc_ORIENTATION_MAX_COUNT__") = "Maximum number of orientations per SIFT feature point";

    auto config_error = nb::exception<popsift::ConfigError>(m, "ConfigError", PyExc_ValueError);
    config_error.doc() = "Raised when SIFT configuration parameters are invalid";
    
    auto invalid_enum_error = nb::exception<popsift::InvalidEnumError>(m, "InvalidEnumError", config_error.ptr());
    invalid_enum_error.doc() = "Raised when an invalid enumeration value is provided";
    
    auto param_range_error = nb::exception<popsift::ParameterRangeError>(m, "ParameterRangeError", config_error.ptr());
    param_range_error.doc() = "Raised when a parameter value is outside its valid range";
    
    auto memory_error = nb::exception<popsift::MemoryError>(m, "MemoryError", PyExc_MemoryError);
    memory_error.doc() = "Raised when memory allocation fails in PopSift operations";
    
    auto cuda_error = nb::exception<popsift::CudaError>(m, "CudaError", PyExc_RuntimeError);
    cuda_error.doc() = "Raised when CUDA operations fail";
    
    auto image_error = nb::exception<popsift::ImageError>(m, "ImageError", PyExc_RuntimeError);
    image_error.doc() = "Raised when image processing operations fail";
    
    auto unsupported_error = nb::exception<popsift::UnsupportedOperationError>(m, "UnsupportedOperationError", PyExc_NotImplementedError);
    unsupported_error.doc() = "Raised when an unsupported operation is attempted";
    
    auto logic_error = nb::exception<popsift::LogicError>(m, "LogicError", PyExc_RuntimeError);
    logic_error.doc() = "Raised when internal logic errors occur";

    nb::set_leak_warnings(false);
    nb::enum_<popsift::Config::GaussMode>(m, "GaussMode")
        .value("VLFeat_Compute", popsift::Config::VLFeat_Compute)
        .value("VLFeat_Relative", popsift::Config::VLFeat_Relative)
        .value("VLFeat_Relative_All", popsift::Config::VLFeat_Relative_All)
        .value("OpenCV_Compute", popsift::Config::OpenCV_Compute)
        .value("Fixed9", popsift::Config::Fixed9)
        .value("Fixed15", popsift::Config::Fixed15)
        .def("__str__", [](popsift::Config::GaussMode mode) {
            switch (mode) {
                case popsift::Config::VLFeat_Compute: return "VLFeat_Compute";
                case popsift::Config::VLFeat_Relative: return "VLFeat_Relative";
                case popsift::Config::VLFeat_Relative_All: return "VLFeat_Relative_All";
                case popsift::Config::OpenCV_Compute: return "OpenCV_Compute";
                case popsift::Config::Fixed9: return "Fixed9";
                case popsift::Config::Fixed15: return "Fixed15";
                default: return "Unknown";
            }
        });

    nb::enum_<popsift::Config::SiftMode>(m, "SiftMode")
        .value("PopSift", popsift::Config::PopSift)
        .value("OpenCV", popsift::Config::OpenCV)
        .value("VLFeat", popsift::Config::VLFeat)
        .def("__str__", [](popsift::Config::SiftMode mode) {
            switch (mode) {
                case popsift::Config::PopSift: return "PopSift";
                case popsift::Config::OpenCV: return "OpenCV";
                case popsift::Config::VLFeat: return "VLFeat";
                default: return "Unknown";
            }
        });

    nb::enum_<popsift::Config::LogMode>(m, "LogMode")
        .value("None", popsift::Config::None)
        .value("All", popsift::Config::All)
        .def("__str__", [](popsift::Config::LogMode mode) {
            return mode == popsift::Config::None ? "None" : "All";
        });

    nb::enum_<popsift::Config::ScalingMode>(m, "ScalingMode")
        .value("ScaleDirect", popsift::Config::ScaleDirect)
        .value("ScaleDefault", popsift::Config::ScaleDefault)
        .def("__str__", [](popsift::Config::ScalingMode mode) {
            return mode == popsift::Config::ScaleDirect ? "ScaleDirect" : "ScaleDefault";
        });

    nb::enum_<popsift::Config::DescMode>(m, "DescMode")
        .value("Loop", popsift::Config::Loop)
        .value("ILoop", popsift::Config::ILoop)
        .value("Grid", popsift::Config::Grid)
        .value("IGrid", popsift::Config::IGrid)
        .value("NoTile", popsift::Config::NoTile)
        .def("__str__", [](popsift::Config::DescMode mode) {
            switch (mode) {
                case popsift::Config::Loop: return "Loop";
                case popsift::Config::ILoop: return "ILoop";
                case popsift::Config::Grid: return "Grid";
                case popsift::Config::IGrid: return "IGrid";
                case popsift::Config::NoTile: return "NoTile";
                default: return "Unknown";
            }
        });

    nb::enum_<popsift::Config::NormMode>(m, "NormMode")
        .value("RootSift", popsift::Config::RootSift)
        .value("Classic", popsift::Config::Classic)
        .def("__str__", [](popsift::Config::NormMode mode) {
            return mode == popsift::Config::RootSift ? "RootSift" : "Classic";
        });

    nb::enum_<popsift::Config::GridFilterMode>(m, "GridFilterMode")
        .value("RandomScale", popsift::Config::RandomScale)
        .value("LargestScaleFirst", popsift::Config::LargestScaleFirst)
        .value("SmallestScaleFirst", popsift::Config::SmallestScaleFirst)
        .def("__str__", [](popsift::Config::GridFilterMode mode) {
            switch (mode) {
                case popsift::Config::RandomScale: return "RandomScale";
                case popsift::Config::LargestScaleFirst: return "LargestScaleFirst";
                case popsift::Config::SmallestScaleFirst: return "SmallestScaleFirst";
                default: return "Unknown";
            }
        });

    nb::enum_<popsift::Config::ProcessingMode>(m, "ProcessingMode")
        .value("ExtractingMode", popsift::Config::ExtractingMode)
        .value("MatchingMode", popsift::Config::MatchingMode)
        .def("__str__", [](popsift::Config::ProcessingMode mode) {
            return mode == popsift::Config::ExtractingMode ? "ExtractingMode" : "MatchingMode";
        });

    nb::class_<popsift::Config>(m, "Config", R"pbdoc(
        SIFT feature extraction configuration.
        
        This class controls all parameters for SIFT feature detection and description.
        It allows fine-tuning of the algorithm for different types of images and
        performance requirements.
        
        Examples:
            Basic usage:
            
            >>> config = Config()
            >>> config.set_sigma(1.6)
            >>> config.set_octaves(4)
            
            Advanced configuration:
            
            >>> config = Config(octaves=5, levels=3, sigma=1.8)
            >>> config.set_gauss_mode("vlfeat")
            >>> config.set_threshold(0.01)
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
            Create a new SIFT configuration with default parameters.
            
            Returns:
                A Config object with the following defaults:
                - octaves: -1 (auto-detect)
                - levels: 3
                - sigma: 1.6
                - threshold: 0.04/3.0
        )pbdoc")
        
        .def("__init__", [](popsift::Config* self, int octaves, int levels, float sigma) {
            new (self) popsift::Config();
            self->setOctaves(octaves);
            self->setLevels(levels);
            self->setSigma(sigma);
        }, nb::arg("octaves") = -1, nb::arg("levels") = 3, nb::arg("sigma") = 1.6f,
        R"pbdoc(
            Create a new SIFT configuration with specified parameters.
            
            Args:
                octaves: Number of octaves (-1 for auto-detection based on image size)
                levels: Number of levels per octave (typically 3)
                sigma: Initial smoothing sigma value (typically 1.6)
                
            Raises:
                ParameterRangeError: If any parameter is outside valid range
        )pbdoc",
        nb::sig("def __init__(self, octaves: int = -1, levels: int = 3, sigma: float = 1.6) -> None"))
        
        .def("set_gauss_mode", static_cast<void(popsift::Config::*)(const std::string&)>(&popsift::Config::setGaussMode),
             R"pbdoc(
                Set Gaussian filtering mode from string.
                
                Args:
                    mode: Gaussian mode string. Valid options:
                        - "vlfeat" (default): VLFeat-style computation
                        - "vlfeat-hw-interpolated": Hardware-interpolated VLFeat
                        - "relative": Synonym for vlfeat-hw-interpolated
                        - "vlfeat-direct": Direct VLFeat computation  
                        - "opencv": OpenCV-style computation
                        - "fixed9": Fixed 9-tap filter
                        - "fixed15": Fixed 15-tap filter
                        
                Raises:
                    InvalidEnumError: If mode string is not recognized
             )pbdoc", 
             nb::arg("mode"),
             nb::sig("def set_gauss_mode(self, mode: typing.Literal['vlfeat', 'vlfeat-hw-interpolated', 'relative', 'vlfeat-direct', 'opencv', 'fixed9', 'fixed15']) -> None"))
        .def("set_gauss_mode", static_cast<void(popsift::Config::*)(popsift::Config::GaussMode)>(&popsift::Config::setGaussMode),
             "Set Gaussian mode using enum value", nb::arg("mode"))
        .def("get_gauss_mode", &popsift::Config::getGaussMode, "Get current Gaussian filtering mode")
        .def_static("get_gauss_mode_default", &popsift::Config::getGaussModeDefault, "Get default Gaussian mode")
        .def_static("get_gauss_mode_usage", &popsift::Config::getGaussModeUsage, "Get help text for all Gaussian mode options")
        
        .def("set_sift_mode", &popsift::Config::setSiftMode, "Set SIFT mode", nb::arg("mode"))
        .def("get_sift_mode", &popsift::Config::getSiftMode, "Get current SIFT mode")
        
        .def("set_log_mode", &popsift::Config::setLogMode, "Set log mode", nb::arg("mode") = popsift::Config::All)
        .def("get_log_mode", &popsift::Config::getLogMode, "Get current log mode")
        
        .def("set_scaling_mode", &popsift::Config::setScalingMode, "Set scaling mode", nb::arg("mode") = popsift::Config::ScaleDefault)
        .def("get_scaling_mode", &popsift::Config::getScalingMode, "Get current scaling mode")
        
        .def("set_desc_mode", static_cast<void(popsift::Config::*)(const std::string&)>(&popsift::Config::setDescMode),
             "Set descriptor mode from string", nb::arg("mode"))
        .def("set_desc_mode", static_cast<void(popsift::Config::*)(popsift::Config::DescMode)>(&popsift::Config::setDescMode),
             "Set descriptor mode", nb::arg("mode") = popsift::Config::Loop)
        .def("get_desc_mode", &popsift::Config::getDescMode, "Get current descriptor mode")
        
        .def("set_norm_mode", static_cast<void(popsift::Config::*)(popsift::Config::NormMode)>(&popsift::Config::setNormMode),
             "Set normalization mode", nb::arg("mode"))
        .def("set_norm_mode", static_cast<void(popsift::Config::*)(const std::string&)>(&popsift::Config::setNormMode),
             "Set normalization mode from string", nb::arg("mode"))
        .def("get_norm_mode", static_cast<popsift::Config::NormMode(popsift::Config::*)() const>(&popsift::Config::getNormMode), "Get current normalization mode")
        .def_static("get_norm_mode_default", &popsift::Config::getNormModeDefault, "Get default normalization mode")
        .def_static("get_norm_mode_usage", &popsift::Config::getNormModeUsage, "Get usage string for normalization modes")
        
        .def("set_filter_sorting", static_cast<void(popsift::Config::*)(const std::string&)>(&popsift::Config::setFilterSorting),
             "Set filter sorting from string", nb::arg("direction"))
        .def("set_filter_sorting", static_cast<void(popsift::Config::*)(popsift::Config::GridFilterMode)>(&popsift::Config::setFilterSorting),
             "Set filter sorting mode", nb::arg("mode"))
        .def("get_filter_sorting", &popsift::Config::getFilterSorting, "Get current filter sorting mode")
        
        .def("set_downsampling", &popsift::Config::setDownsampling, "Set downsampling factor", nb::arg("factor"))
        .def("set_octaves", &popsift::Config::setOctaves, "Set number of octaves", nb::arg("octaves"))
        .def("set_levels", &popsift::Config::setLevels, "Set number of levels per octave", nb::arg("levels"))
        .def("set_sigma", &popsift::Config::setSigma, "Set sigma value", nb::arg("sigma"))
        .def("set_edge_limit", &popsift::Config::setEdgeLimit, "Set edge limit", nb::arg("limit"))
        .def("set_threshold", &popsift::Config::setThreshold, "Set threshold", nb::arg("threshold"))
        .def("set_initial_blur", &popsift::Config::setInitialBlur, "Set initial blur", nb::arg("blur"))
        .def("set_filter_max_extrema", &popsift::Config::setFilterMaxExtrema, "Set maximum number of extrema to filter", nb::arg("extrema"))
        .def("set_filter_grid_size", &popsift::Config::setFilterGridSize, "Set grid size for filtering", nb::arg("size"))
        .def("set_normalization_multiplier", &popsift::Config::setNormalizationMultiplier, "Set normalization multiplier", nb::arg("multiplier"))
        .def("set_verbose", &popsift::Config::setVerbose, "Set verbose mode", nb::arg("enabled") = true)
        .def("set_print_gauss_tables", &popsift::Config::setPrintGaussTables, "Enable printing of Gaussian tables")
        
        .def("get_upscale_factor", &popsift::Config::getUpscaleFactor, "Get upscale factor")
        .def("get_max_extrema", &popsift::Config::getMaxExtrema, "Get maximum number of extrema")
        .def("get_filter_max_extrema", &popsift::Config::getFilterMaxExtrema, "Get maximum number of extrema for filtering")
        .def("get_filter_grid_size", &popsift::Config::getFilterGridSize, "Get grid size for filtering")
        .def("get_normalization_multiplier", &popsift::Config::getNormalizationMultiplier, "Get normalization multiplier")
        .def("get_peak_threshold", &popsift::Config::getPeakThreshold, "Get computed peak threshold")
        .def("get_can_filter_extrema", &popsift::Config::getCanFilterExtrema, "Check if extrema filtering is available")
        .def("has_initial_blur", &popsift::Config::hasInitialBlur, "Check if initial blur is enabled")
        .def("get_initial_blur", &popsift::Config::getInitialBlur, "Get initial blur value")
        .def("if_print_gauss_tables", &popsift::Config::ifPrintGaussTables, "Check if Gaussian tables should be printed")
        
        .def_rw("octaves", &popsift::Config::octaves, "Number of octaves (-1 for auto)")
        .def_rw("levels", &popsift::Config::levels, "Number of levels per octave")
        .def_rw("sigma", &popsift::Config::sigma, "Sigma value")
        .def_rw("verbose", &popsift::Config::verbose, "Verbose output flag")
        
        .def("__eq__", &popsift::Config::equal, nb::arg("other"))
        .def("__ne__", [](const popsift::Config& self, const popsift::Config& other) { return !self.equal(other); }, nb::arg("other"))
        
        .def("__repr__", [](const popsift::Config& self) {
            std::ostringstream oss;
            oss << "Config(octaves=" << self.octaves 
                << ", levels=" << self.levels 
                << ", sigma=" << self.sigma 
                << ", verbose=" << (self.verbose ? "True" : "False") << ")";
            return oss.str();
        })
        .def("__str__", [](const popsift::Config& self) {
            std::ostringstream oss;
            oss << "SIFT Configuration:\n"
                << "  Octaves: " << self.octaves << "\n"
                << "  Levels: " << self.levels << "\n"
                << "  Sigma: " << self.sigma << "\n"
                << "  Gaussian Mode: " << static_cast<int>(self.getGaussMode()) << "\n"
                << "  SIFT Mode: " << static_cast<int>(self.getSiftMode()) << "\n"
                << "  Descriptor Mode: " << static_cast<int>(self.getDescMode()) << "\n"
                << "  Normalization Mode: " << static_cast<int>(self.getNormMode()) << "\n"
                << "  Filter Max Extrema: " << self.getFilterMaxExtrema() << "\n"
                << "  Filter Grid Size: " << self.getFilterGridSize() << "\n"
                << "  Verbose: " << (self.verbose ? "True" : "False");
            return oss.str();
        });

    nb::class_<popsift::Descriptor>(m, "Descriptor")
        .def(nb::init<>(), "Create an empty SIFT descriptor with 128 features")
        .def_prop_ro("features", [](popsift::Descriptor& self) {
            return nb::ndarray<float, nb::numpy, nb::shape<128>>(
                self.features
            );
        }, nb::rv_policy::reference_internal, "Get the 128-dimensional SIFT feature vector as a NumPy array")
        .def("__len__", [](const popsift::Descriptor& self) {
            return 128;
        }, "Return the number of features (always 128)")
        .def("__getitem__", [](popsift::Descriptor& self, int index) -> float {
            if (index < 0 || index >= 128) {
                throw std::out_of_range("Descriptor index out of range (0-127)");
            }
            return self.features[index];
        }, "Get a feature value by index")
        .def("__setitem__", [](popsift::Descriptor& self, int index, float value) {
            if (index < 0 || index >= 128) {
                throw std::out_of_range("Descriptor index out of range (0-127)");
            }
            self.features[index] = value;
        }, "Set a feature value by index")
        .def("__repr__", [](const popsift::Descriptor& self) {
            return "<Descriptor with 128 features>";
        })
        .def("__str__", [](const popsift::Descriptor& self) {
            std::ostringstream oss;
            oss << "Descriptor([";
            for (int i = 0; i < 128; i++) {
                if (i > 0) oss << ", ";
                oss << self.features[i];
            }
            oss << "])";
            return oss.str();
        });

    nb::class_<popsift::Feature>(m, "Feature")
        .def(nb::init<>(), "Create an empty SIFT feature")
        .def_ro("debug_octave", &popsift::Feature::debug_octave, "Debug octave information")
        .def_ro("xpos", &popsift::Feature::xpos, "X-coordinate of the feature (scale-adapted)")
        .def_ro("ypos", &popsift::Feature::ypos, "Y-coordinate of the feature (scale-adapted)")
        .def_ro("sigma", &popsift::Feature::sigma, "Scale (sigma) of the feature")
        .def_ro("num_ori", &popsift::Feature::num_ori, "Number of orientations for this feature")
        .def_prop_ro("orientation", [](popsift::Feature& self) {
            return nb::ndarray<float, nb::numpy, nb::shape<ORIENTATION_MAX_COUNT>>(
                self.orientation
            );
        }, nb::rv_policy::reference_internal, "Get all orientations as a NumPy array")
        .def("get_descriptor", [](popsift::Feature& self, int index) -> popsift::Descriptor* {
            if (index < 0 || index >= self.num_ori || index >= ORIENTATION_MAX_COUNT) {
                throw std::out_of_range("Descriptor index out of range");
            }
            if (!self.desc[index]) {
                throw std::runtime_error("Descriptor at index " + std::to_string(index) + " is null");
            }
            return self.desc[index];
        }, nb::rv_policy::reference_internal, "Get descriptor by index")
        .def("get_orientation", [](popsift::Feature& self, int index) -> float {
            if (index < 0 || index >= self.num_ori || index >= ORIENTATION_MAX_COUNT) {
                throw std::out_of_range("Orientation index out of range");
            }
            return self.orientation[index];
        }, "Get orientation by index")
        .def("__len__", [](const popsift::Feature& self) {
            return self.num_ori;
        }, "Return the number of orientations/descriptors")
        .def("__getitem__", [](popsift::Feature& self, int index) -> popsift::Descriptor* {
            if (index < 0 || index >= self.num_ori || index >= ORIENTATION_MAX_COUNT) {
                throw std::out_of_range("Descriptor index out of range");
            }
            if (!self.desc[index]) {
                throw std::runtime_error("Descriptor at index " + std::to_string(index) + " is null");
            }
            return self.desc[index];
        }, nb::rv_policy::reference_internal, "Get descriptor by index (array-style access)")
        .def("__iter__", [](popsift::Feature& self) {
            std::vector<popsift::Descriptor*> valid_descriptors;
            for (int i = 0; i < self.num_ori && i < ORIENTATION_MAX_COUNT; i++) {
                if (self.desc[i]) {
                    valid_descriptors.push_back(self.desc[i]);
                }
            }
            return nb::make_iterator(nb::type<popsift::Feature>(), "FeatureDescriptorIterator",
                                   valid_descriptors.begin(), valid_descriptors.end());
        }, nb::keep_alive<0, 1>(), "Iterate over valid descriptors")
        .def("__repr__", [](const popsift::Feature& self) {
            std::ostringstream oss;
            self.print(oss, false);
            return oss.str();
        })
        .def("__str__", [](const popsift::Feature& self) {
            std::ostringstream oss;
            oss << "Feature(x=" << self.xpos << ", y=" << self.ypos 
                << ", sigma=" << self.sigma << ", orientations=" << self.num_ori << ")";
            return oss.str();
        });

    nb::class_<popsift::FeaturesHost>(m, "FeaturesHost")
        .def(nb::init<>(), "Create an empty features container")
        .def(nb::init<int, int>(), "Create a features container with specified feature and descriptor counts")
        .def("size", &popsift::FeaturesHost::size, "Get the number of features")
        .def("get_feature_count", &popsift::FeaturesHost::getFeatureCount, "Get the number of features")
        .def("get_descriptor_count", &popsift::FeaturesHost::getDescriptorCount, "Get the number of descriptors")
        .def("reset", &popsift::FeaturesHost::reset, "Reset the container with new feature and descriptor counts")
        .def("pin", &popsift::FeaturesHost::pin, "Pin memory for CUDA operations")
        .def("unpin", &popsift::FeaturesHost::unpin, "Unpin memory after CUDA operations")
        .def("__len__", &popsift::FeaturesHost::size, "Get the number of features")
        .def("__getitem__", [](popsift::FeaturesHost& self, int index) -> popsift::Feature* {
            if (index < 0 || index >= self.size()) {
                throw std::out_of_range("Feature index out of range");
            }
            return const_cast<popsift::Feature*>(&self.begin()[index]);
        }, nb::rv_policy::reference_internal, "Get feature by index")
        .def("__iter__", [](popsift::FeaturesHost& self) {
            return nb::make_iterator(nb::type<popsift::FeaturesHost>(), "FeaturesIterator",
                                   self.begin(), self.end());
        }, nb::keep_alive<0, 1>(), "Iterate over features")
        .def("__repr__", [](const popsift::FeaturesHost& self) {
            std::ostringstream oss;
            oss << "<FeaturesHost with " << self.getFeatureCount() 
                << " features and " << self.getDescriptorCount() << " descriptors>";
            return oss.str();
        })
        .def("__str__", [](const popsift::FeaturesHost& self) {
            std::ostringstream oss;
            oss << "FeaturesHost(features=" << self.getFeatureCount() 
                << ", descriptors=" << self.getDescriptorCount() << ")";
            return oss.str();
        });

    m.attr("Features") = m.attr("FeaturesHost");

    // FeaturesDev class binding with CuPy integration
    nb::class_<popsift::FeaturesDev>(m, "FeaturesDev", R"pbdoc(
        Container for SIFT features and descriptors in GPU device memory.
        
        FeaturesDev holds SIFT features and descriptors in CUDA device memory,
        optimized for GPU-to-GPU matching operations. It provides CuPy array
        views of the device memory for easy integration with GPU computing workflows.
        
        Examples:
            Basic usage:
            
            >>> features_dev = FeaturesDev(100, 200)
            >>> features_array = features_dev.get_features_array()  # CuPy array
            >>> descriptors_array = features_dev.get_descriptors_array()  # CuPy array
            
            GPU matching workflow:
            
            >>> # Extract features in matching mode
            >>> config = Config()
            >>> config.set_processing_mode(ProcessingMode.MatchingMode)
            >>> popsift = PopSift(config, ProcessingMode.MatchingMode)
            >>> job = popsift.enqueue(width, height, image_data)
            >>> features_dev = job.get_dev()
            >>> # Use CuPy arrays for GPU operations
            >>> features_cp = features_dev.get_features_array()
            >>> descriptors_cp = features_dev.get_descriptors_array()
    )pbdoc")
        .def(nb::init<>(), "Create an empty features container in device memory")
        .def(nb::init<int, int>(), "Create a features container with specified feature and descriptor counts in device memory")
        .def("size", &popsift::FeaturesDev::size, "Get the number of features")
        .def("get_feature_count", &popsift::FeaturesDev::getFeatureCount, "Get the number of features")
        .def("get_descriptor_count", &popsift::FeaturesDev::getDescriptorCount, "Get the number of descriptors")
        .def("reset", &popsift::FeaturesDev::reset, "Reset the container with new feature and descriptor counts")
        .def("match", &popsift::FeaturesDev::match, "Match features against another FeaturesDev object", nb::arg("other"))
        .def("get_features_array", [](popsift::FeaturesDev& self) {
            nb::module_ cupy = nb::module_::import_("cupy");
            void* ptr = self.getFeatures();
            int feature_count = self.getFeatureCount();
            size_t bytes = feature_count * sizeof(popsift::Feature);
            int device = get_cuda_device();
            nb::object unowned = cupy.attr("cuda").attr("UnownedMemory")(
                reinterpret_cast<uintptr_t>(ptr), bytes, nb::cast(&self), device
            );
            nb::object memptr = cupy.attr("cuda").attr("MemoryPointer")(unowned, 0);
            nb::object array = cupy.attr("ndarray")(
                nb::make_tuple(feature_count, 7),
                cupy.attr("float32"),
                memptr
            );
            return array;
        }, nb::rv_policy::reference_internal, R"pbdoc(
            Get features as a zero-copy CuPy array in device memory.
            
            Returns:
                cupy.ndarray: CuPy array view of features in device memory.
                             Shape is (num_features, 7) where the 7 fields are:
                             [debug_octave, xpos, ypos, sigma, num_ori, orientation[0], orientation[1]]
                             
            Note:
                This returns a view into the device memory, not a copy.
                The array is only valid while the FeaturesDev object exists.
        )pbdoc")
        .def("get_descriptors_array", [](popsift::FeaturesDev& self) {
            nb::module_ cupy = nb::module_::import_("cupy");
            void* ptr = self.getDescriptors();
            int descriptor_count = self.getDescriptorCount();
            size_t bytes = descriptor_count * sizeof(popsift::Descriptor);
            int device = get_cuda_device();
            nb::object unowned = cupy.attr("cuda").attr("UnownedMemory")(
                reinterpret_cast<uintptr_t>(ptr), bytes, nb::cast(&self), device
            );
            nb::object memptr = cupy.attr("cuda").attr("MemoryPointer")(unowned, 0);
            nb::object array = cupy.attr("ndarray")(
                nb::make_tuple(descriptor_count, 128),
                cupy.attr("float32"),
                memptr
            );
            return array;
        }, nb::rv_policy::reference_internal, R"pbdoc(
            Get descriptors as a zero-copy CuPy array in device memory.
            
            Returns:
                cupy.ndarray: CuPy array view of descriptors in device memory.
                             Shape is (num_descriptors, 128) for SIFT features.
                             
            Note:
                This returns a view into the device memory, not a copy.
                The array is only valid while the FeaturesDev object exists.
        )pbdoc")
        .def("get_reverse_map_array", [](popsift::FeaturesDev& self) {
            nb::module_ cupy = nb::module_::import_("cupy");
            void* ptr = self.getReverseMap();
            int descriptor_count = self.getDescriptorCount();
            size_t bytes = descriptor_count * sizeof(int);
            int device = get_cuda_device();
            nb::object unowned = cupy.attr("cuda").attr("UnownedMemory")(
                reinterpret_cast<uintptr_t>(ptr), bytes, nb::cast(&self), device
            );
            nb::object memptr = cupy.attr("cuda").attr("MemoryPointer")(unowned, 0);
            nb::object array = cupy.attr("ndarray")(
                nb::make_tuple(descriptor_count),
                cupy.attr("int32"),
                memptr
            );
            return array;
        }, nb::rv_policy::reference_internal, R"pbdoc(
            Get reverse mapping as a zero-copy CuPy array in device memory.
            
            Returns:
                cupy.ndarray: CuPy array view of reverse mapping in device memory.
                             Shape is (num_descriptors,) mapping descriptors to features.
                             
            Note:
                This returns a view into the device memory, not a copy.
                The array is only valid while the FeaturesDev object exists.
        )pbdoc")
        .def("to_host", [](popsift::FeaturesDev& self) -> popsift::FeaturesHost* {
            // Create a new FeaturesHost with the same dimensions
            int feature_count = self.getFeatureCount();
            int descriptor_count = self.getDescriptorCount();
            
            auto* host_features = new popsift::FeaturesHost(feature_count, descriptor_count);
            
            // Copy data from device to host
            cudaMemcpy(host_features->getFeatures(), self.getFeatures(), 
                      feature_count * sizeof(popsift::Feature), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_features->getDescriptors(), self.getDescriptors(), 
                      descriptor_count * sizeof(popsift::Descriptor), cudaMemcpyDeviceToHost);
            
            return host_features;
        }, nb::rv_policy::take_ownership, R"pbdoc(
            Convert device features to host features.
            
            Returns:
                FeaturesHost: New FeaturesHost object with data copied from device to host.
                             
            Note:
                This performs a device-to-host memory copy and creates a new object.
                The returned object is independent of the original FeaturesDev.
        )pbdoc")
        .def("__len__", &popsift::FeaturesDev::size, "Get the number of features")
        .def("__repr__", [](const popsift::FeaturesDev& self) {
            std::ostringstream oss;
            oss << "<FeaturesDev with " << self.getFeatureCount() 
                << " features and " << self.getDescriptorCount() << " descriptors in device memory>";
            return oss.str();
        })
        .def("__str__", [](const popsift::FeaturesDev& self) {
            std::ostringstream oss;
            oss << "FeaturesDev(features=" << self.getFeatureCount() 
                << ", descriptors=" << self.getDescriptorCount() << ", device_memory=True)";
            return oss.str();
        });

    // Add PopSift ImageMode enum
    nb::enum_<PopSift::ImageMode>(m, "ImageMode")
        .value("ByteImages", PopSift::ByteImages, "Byte images with value range 0..255")
        .value("FloatImages", PopSift::FloatImages, "Float images with value range [0..1[")
        .def("__str__", [](PopSift::ImageMode mode) {
            return mode == PopSift::ByteImages ? "ByteImages" : "FloatImages";
        });

    // Add PopSift AllocTest enum
    nb::enum_<PopSift::AllocTest>(m, "AllocTest")
        .value("Ok", PopSift::Ok, "Image dimensions are supported by this device's CUDA texture engine")
        .value("ImageExceedsLinearTextureLimit", PopSift::ImageExceedsLinearTextureLimit, 
               "Input image size exceeds the dimensions of the CUDA Texture used for loading")
        .value("ImageExceedsLayeredSurfaceLimit", PopSift::ImageExceedsLayeredSurfaceLimit,
               "Scaled input image exceeds the dimensions of the CUDA Surface used for the image pyramid")
        .def("__str__", [](PopSift::AllocTest test) {
            switch (test) {
                case PopSift::Ok: return "Ok";
                case PopSift::ImageExceedsLinearTextureLimit: return "ImageExceedsLinearTextureLimit";
                case PopSift::ImageExceedsLayeredSurfaceLimit: return "ImageExceedsLayeredSurfaceLimit";
                default: return "Unknown";
            }
        });

    // SiftJob class binding
    nb::class_<SiftJob>(m, "SiftJob", R"pbdoc(
        A job for processing an image through the PopSift pipeline.
        
        SiftJob represents an asynchronous task for extracting SIFT features from an image.
        It provides a future-like interface to retrieve the results once processing is complete.
        
        Example:
            >>> job = popsift.enqueue(width, height, image_data)
            >>> features = job.get()  # Wait for completion and get results
    )pbdoc")
        .def("get", [](SiftJob& self) -> popsift::FeaturesHost* {
            return self.get();
        }, nb::rv_policy::reference_internal, R"pbdoc(
            Wait for job completion and return the extracted features.
            
            Returns:
                FeaturesHost: The extracted SIFT features
                
            Raises:
                RuntimeError: If the job failed to process
        )pbdoc")
        .def("get_host", [](SiftJob& self) -> popsift::FeaturesHost* {
            return self.getHost();
        }, nb::rv_policy::reference_internal, "Get features as host memory (alias for get())")
        .def("get_dev", [](SiftJob& self) -> popsift::FeaturesDev* {
            return self.getDev();
        }, nb::rv_policy::reference_internal, R"pbdoc(
            Get features as device memory (for matching mode).
            
            Returns:
                FeaturesDev: The extracted SIFT features in device memory
                
            Raises:
                RuntimeError: If the job failed to process or is not in matching mode
        )pbdoc")
        .def("__repr__", [](const SiftJob& self) {
            return "<SiftJob>";
        })
        .def("__str__", [](const SiftJob& self) {
            return "SiftJob";
        });

    // PopSift class binding
    nb::class_<PopSift>(m, "PopSift", R"pbdoc(
        Main PopSift pipeline for SIFT feature extraction.
        
        PopSift provides a high-performance, GPU-accelerated SIFT feature extraction
        pipeline. It supports both byte (0-255) and float (0-1) image formats and
        can operate in extracting mode (downloads features to host) or matching mode
        (keeps features in device memory for fast matching).
        
        Examples:
            Basic usage with byte images:
            
            >>> popsift = PopSift(ImageMode.ByteImages)
            >>> job = popsift.enqueue(width, height, image_data)
            >>> features = job.get()
            
            Advanced usage with configuration:
            
            >>> config = Config(octaves=4, levels=3, sigma=1.6)
            >>> popsift = PopSift(config, ProcessingMode.ExtractingMode, ImageMode.ByteImages)
            >>> job = popsift.enqueue(width, height, image_data)
            >>> features = job.get()
    )pbdoc")
        .def(nb::init<PopSift::ImageMode, int>(), 
             nb::arg("image_mode") = PopSift::ByteImages, 
             nb::arg("device") = 0,
             R"pbdoc(
                Create a PopSift pipeline with default configuration.
                
                Args:
                    image_mode: Type of images to process (ByteImages or FloatImages)
                    device: CUDA device ID to use (default: 0)
             )pbdoc")
        .def(nb::init<const popsift::Config&, popsift::Config::ProcessingMode, PopSift::ImageMode, int>(),
             nb::arg("config"),
             nb::arg("mode") = popsift::Config::ExtractingMode,
             nb::arg("image_mode") = PopSift::ByteImages,
             nb::arg("device") = 0,
             R"pbdoc(
                Create a PopSift pipeline with custom configuration.
                
                Args:
                    config: SIFT configuration parameters
                    mode: Processing mode (ExtractingMode or MatchingMode)
                    image_mode: Type of images to process
                    device: CUDA device ID to use
             )pbdoc")
        .def("configure", &PopSift::configure, 
             nb::arg("config"), nb::arg("force") = false,
             R"pbdoc(
                Configure the pipeline with new parameters.
                
                Args:
                    config: New SIFT configuration
                    force: Force reconfiguration even if parameters haven't changed
                    
                Returns:
                    bool: True if configuration was applied successfully
             )pbdoc")
        .def("uninit", static_cast<void(PopSift::*)()>(&PopSift::uninit),
             R"pbdoc(
                Release all allocated resources.
                
                This should be called when the pipeline is no longer needed to free
                GPU memory and other resources.
             )pbdoc")
        .def("test_texture_fit", &PopSift::testTextureFit,
             nb::arg("width"), nb::arg("height"),
             R"pbdoc(
                Test if the current CUDA device can support the given image dimensions.
                
                Args:
                    width: Image width in pixels
                    height: Image height in pixels
                    
                Returns:
                    AllocTest: Result indicating if the image size is supported
             )pbdoc")
        .def("test_texture_fit_error_string", &PopSift::testTextureFitErrorString,
             nb::arg("err"), nb::arg("width"), nb::arg("height"),
             R"pbdoc(
                Get a descriptive error message for texture fit test results.
                
                Args:
                    err: AllocTest result from test_texture_fit
                    width: Original image width
                    height: Original image height
                    
                Returns:
                    str: Human-readable error message with suggestions
             )pbdoc")
        .def("enqueue", [](PopSift& self, int w, int h, const nb::ndarray<uint8_t, nb::numpy>& image_data) -> SiftJob* {
            if (image_data.ndim() != 2) {
                throw std::invalid_argument("Image data must be 2D array");
            }
            if (image_data.shape(0) != h || image_data.shape(1) != w) {
                throw std::invalid_argument("Image dimensions don't match width/height parameters");
            }
            return self.enqueue(w, h, image_data.data());
        }, nb::arg("width"), nb::arg("height"), nb::arg("image_data"),
           R"pbdoc(
                Enqueue a byte image for SIFT feature extraction.
                
                Args:
                    width: Image width in pixels
                    height: Image height in pixels
                    image_data: 2D numpy array with uint8 values (0-255)
                    
                Returns:
                    SiftJob: Job object for tracking the processing task
                    
                Raises:
                    ImageError: If image mode is not ByteImages
                    RuntimeError: If image dimensions exceed GPU limits
             )pbdoc")
        .def("enqueue", [](PopSift& self, int w, int h, const nb::ndarray<float, nb::numpy>& image_data) -> SiftJob* {
            if (image_data.ndim() != 2) {
                throw std::invalid_argument("Image data must be 2D array");
            }
            if (image_data.shape(0) != h || image_data.shape(1) != w) {
                throw std::invalid_argument("Image dimensions don't match width/height parameters");
            }
            return self.enqueue(w, h, image_data.data());
        }, nb::arg("width"), nb::arg("height"), nb::arg("image_data"),
           R"pbdoc(
                Enqueue a float image for SIFT feature extraction.
                
                Args:
                    width: Image width in pixels
                    height: Image height in pixels
                    image_data: 2D numpy array with float values (0.0-1.0)
                    
                Returns:
                    SiftJob: Job object for tracking the processing task
                    
                Raises:
                    ImageError: If image mode is not FloatImages
                    RuntimeError: If image dimensions exceed GPU limits
             )pbdoc")
        .def("__repr__", [](const PopSift& self) {
            return "<PopSift>";
        })
        .def("__str__", [](const PopSift& self) {
            return "PopSift";
        });
}
