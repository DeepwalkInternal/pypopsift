#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>  
#include <nanobind/make_iterator.h>
#include <nanobind/stl/string.h>
#include <popsift/sift_extremum.h>
#include <popsift/features.h>
#include <popsift/sift_constants.h>
#include <popsift/sift_conf.h>
#include <sstream>

namespace nb = nanobind;

NB_MODULE(pypopsift, m) {
    m.doc() = "Python bindings for the PopSift library, a real-time SIFT implementation in CUDA.";

    // Expose the constant for Python users
    m.attr("ORIENTATION_MAX_COUNT") = ORIENTATION_MAX_COUNT;

    // Bind enums first
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

    // Bind the Config class
    nb::class_<popsift::Config>(m, "Config")
        .def(nb::init<>(), "Create a new SIFT configuration with default parameters")
        
        // Constructor with common parameters
        .def("__init__", [](popsift::Config* self, int octaves, int levels, float sigma) {
            new (self) popsift::Config();
            self->setOctaves(octaves);
            self->setLevels(levels);
            self->setSigma(sigma);
        }, nb::arg("octaves") = -1, nb::arg("levels") = 3, nb::arg("sigma") = 1.6f,
        "Create a new SIFT configuration with specified octaves, levels, and sigma")
        
        // Gaussian mode methods
        .def("set_gauss_mode", static_cast<void(popsift::Config::*)(const std::string&)>(&popsift::Config::setGaussMode),
             "Set Gaussian mode from string", nb::arg("mode"))
        .def("set_gauss_mode", static_cast<void(popsift::Config::*)(popsift::Config::GaussMode)>(&popsift::Config::setGaussMode),
             "Set Gaussian mode", nb::arg("mode"))
        .def("get_gauss_mode", &popsift::Config::getGaussMode, "Get current Gaussian mode")
        .def_static("get_gauss_mode_default", &popsift::Config::getGaussModeDefault, "Get default Gaussian mode")
        .def_static("get_gauss_mode_usage", &popsift::Config::getGaussModeUsage, "Get usage string for Gaussian modes")
        
        // SIFT mode methods
        .def("set_mode", &popsift::Config::setMode, "Set SIFT mode", nb::arg("mode"))
        .def("get_sift_mode", &popsift::Config::getSiftMode, "Get current SIFT mode")
        
        // Log mode methods
        .def("set_log_mode", &popsift::Config::setLogMode, "Set log mode", nb::arg("mode") = popsift::Config::All)
        .def("get_log_mode", &popsift::Config::getLogMode, "Get current log mode")
        
        // Scaling mode methods
        .def("set_scaling_mode", &popsift::Config::setScalingMode, "Set scaling mode", nb::arg("mode") = popsift::Config::ScaleDefault)
        .def("get_scaling_mode", &popsift::Config::getScalingMode, "Get current scaling mode")
        
        // Descriptor mode methods
        .def("set_desc_mode", static_cast<void(popsift::Config::*)(const std::string&)>(&popsift::Config::setDescMode),
             "Set descriptor mode from string", nb::arg("mode"))
        .def("set_desc_mode", static_cast<void(popsift::Config::*)(popsift::Config::DescMode)>(&popsift::Config::setDescMode),
             "Set descriptor mode", nb::arg("mode") = popsift::Config::Loop)
        .def("get_desc_mode", &popsift::Config::getDescMode, "Get current descriptor mode")
        
        // Normalization mode methods
        .def("set_norm_mode", static_cast<void(popsift::Config::*)(popsift::Config::NormMode)>(&popsift::Config::setNormMode),
             "Set normalization mode", nb::arg("mode"))
        .def("set_norm_mode", static_cast<void(popsift::Config::*)(const std::string&)>(&popsift::Config::setNormMode),
             "Set normalization mode from string", nb::arg("mode"))
        .def("get_norm_mode", static_cast<popsift::Config::NormMode(popsift::Config::*)() const>(&popsift::Config::getNormMode), "Get current normalization mode")
        .def_static("get_norm_mode_default", &popsift::Config::getNormModeDefault, "Get default normalization mode")
        .def_static("get_norm_mode_usage", &popsift::Config::getNormModeUsage, "Get usage string for normalization modes")
        
        // Filter mode methods
        .def("set_filter_sorting", static_cast<void(popsift::Config::*)(const std::string&)>(&popsift::Config::setFilterSorting),
             "Set filter sorting from string", nb::arg("direction"))
        .def("set_filter_sorting", static_cast<void(popsift::Config::*)(popsift::Config::GridFilterMode)>(&popsift::Config::setFilterSorting),
             "Set filter sorting mode", nb::arg("mode"))
        .def("get_filter_sorting", &popsift::Config::getFilterSorting, "Get current filter sorting mode")
        
        // Parameter setters
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
        .def("set_verbose", &popsift::Config::setVerbose, "Set verbose mode", nb::arg("on") = true)
        .def("set_print_gauss_tables", &popsift::Config::setPrintGaussTables, "Enable printing of Gaussian tables")
        
        // Parameter getters
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
        

        
        // Public fields (read-write)
        .def_rw("octaves", &popsift::Config::octaves, "Number of octaves (-1 for auto)")
        .def_rw("levels", &popsift::Config::levels, "Number of levels per octave")
        .def_rw("sigma", &popsift::Config::sigma, "Sigma value")
        .def_rw("verbose", &popsift::Config::verbose, "Verbose output flag")
        
        // Comparison operators
        .def("__eq__", &popsift::Config::equal, nb::arg("other"))
        .def("__ne__", [](const popsift::Config& self, const popsift::Config& other) { return !self.equal(other); }, nb::arg("other"))
        
        // String representation
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
}
