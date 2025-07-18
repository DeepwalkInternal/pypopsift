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
#include <cuda_runtime.h>

namespace nb = nanobind;

void bind_config(nb::module_& m) {
    auto config = nb::class_<popsift::Config>(m, "Config");

    nb::enum_<popsift::Config::GaussMode>(config, "GaussMode")
        .value("VLFeat_Compute", popsift::Config::VLFeat_Compute)
        .value("VLFeat_Relative", popsift::Config::VLFeat_Relative)
        .value("VLFeat_Relative_All", popsift::Config::VLFeat_Relative_All)
        .value("OpenCV_Compute", popsift::Config::OpenCV_Compute)
        .value("Fixed9", popsift::Config::Fixed9)
        .value("Fixed15", popsift::Config::Fixed15)
        .export_values();

    nb::enum_<popsift::Config::SiftMode>(config, "SiftMode")
        .value("PopSift", popsift::Config::PopSift)
        .value("OpenCV", popsift::Config::OpenCV)
        .value("VLFeat", popsift::Config::VLFeat)
        .value("Default", popsift::Config::Default)
        .export_values();

    nb::enum_<popsift::Config::LogMode>(config, "LogMode")
        .value("None", popsift::Config::LogMode::None) // Use scope resolution for None
        .value("All", popsift::Config::All)
        .export_values();

    nb::enum_<popsift::Config::ScalingMode>(config, "ScalingMode")
        .value("ScaleDirect", popsift::Config::ScaleDirect)
        .value("ScaleDefault", popsift::Config::ScaleDefault)
        .export_values();

    nb::enum_<popsift::Config::DescMode>(config, "DescMode")
        .value("Loop", popsift::Config::Loop)
        .value("ILoop", popsift::Config::ILoop)
        .value("Grid", popsift::Config::Grid)
        .value("IGrid", popsift::Config::IGrid)
        .value("NoTile", popsift::Config::NoTile)
        .export_values();

    nb::enum_<popsift::Config::NormMode>(config, "NormMode")
        .value("RootSift", popsift::Config::RootSift)
        .value("Classic", popsift::Config::Classic)
        .export_values();
        
    nb::enum_<popsift::Config::GridFilterMode>(config, "GridFilterMode")
        .value("RandomScale", popsift::Config::RandomScale)
        .value("LargestScaleFirst", popsift::Config::LargestScaleFirst)
        .value("SmallestScaleFirst", popsift::Config::SmallestScaleFirst)
        .export_values();

    nb::enum_<popsift::Config::ProcessingMode>(config, "ProcessingMode")
        .value("ExtractingMode", popsift::Config::ExtractingMode)
        .value("MatchingMode", popsift::Config::MatchingMode)
        .export_values();

    config
        .def(nb::init<>())
        .def_prop_rw(
            "upscale_factor", 
            &popsift::Config::getUpscaleFactor, 
            &popsift::Config::setUpscaleFactor
        )
        .def_prop_rw(
            "octaves",
            &popsift::Config::getOctaves,
            &popsift::Config::setOctaves
        )
        .def_prop_rw(
            "levels",
            &popsift::Config::getLevels,
            &popsift::Config::setLevels
        )
        .def_prop_rw(
            "sigma",
            &popsift::Config::getSigma,
            &popsift::Config::setSigma
        )
        .def_prop_rw(
            "edge_limit", 
            &popsift::Config::getEdgeLimit, 
            &popsift::Config::setEdgeLimit
        )
        .def_prop_rw(
            "threshold",
            &popsift::Config::getThreshold,
            &popsift::Config::setThreshold
        )
        .def_prop_rw(
            "gauss_mode", 
            &popsift::Config::getGaussMode, 
            nb::overload_cast<popsift::Config::GaussMode>(&popsift::Config::setGaussMode)
        )
        .def_prop_rw(
            "sift_mode", 
            &popsift::Config::getSiftMode, 
            &popsift::Config::setSiftMode
        )
        .def_prop_rw(
            "log_mode", 
            &popsift::Config::getLogMode, 
            &popsift::Config::setLogMode
        )
        .def_prop_rw(
            "scaling_mode", 
            &popsift::Config::getScalingMode, 
            &popsift::Config::setScalingMode
        )
        .def_prop_rw(
            "desc_mode", 
            &popsift::Config::getDescMode, 
            nb::overload_cast<popsift::Config::DescMode>(&popsift::Config::setDescMode)
        )
        .def_prop_rw(
            "grid_filter_mode", 
            &popsift::Config::getFilterSorting, 
            nb::overload_cast<popsift::Config::GridFilterMode>(&popsift::Config::setFilterSorting)
        )
        .def_ro("verbose", &popsift::Config::verbose)
        .def("set_verbose", &popsift::Config::setVerbose)
        .def_prop_rw(
            "filter_max_extrema",
            &popsift::Config::getFilterMaxExtrema,
            &popsift::Config::setFilterMaxExtrema
        )
        .def_prop_rw(
            "filter_grid_size",
            &popsift::Config::getFilterGridSize,
            &popsift::Config::setFilterGridSize
        )
        .def_prop_rw(
            "initial_blur", 
            &popsift::Config::getInitialBlur, 
            &popsift::Config::setInitialBlur
        )
        .def_prop_rw(
            "norm_mode", 
            &popsift::Config::getNormMode, 
            nb::overload_cast<popsift::Config::NormMode>(&popsift::Config::setNormMode)
        )
        .def_prop_rw(
            "normalization_multiplier",
            &popsift::Config::getNormalizationMultiplier,
            &popsift::Config::setNormalizationMultiplier
        )
        .def_prop_rw(
            "print_gauss_tables",
            &popsift::Config::ifPrintGaussTables,
            &popsift::Config::setPrintGaussTables
        )
        .def_prop_ro("peak_threshold", &popsift::Config::getPeakThreshold)
        .def_prop_ro("max_extrema", &popsift::Config::getMaxExtrema)
        .def_prop_ro("has_initial_blur", &popsift::Config::hasInitialBlur)
        .def_prop_ro("can_filter_extrema", &popsift::Config::getCanFilterExtrema)

        .def("__eq__", &popsift::Config::equal);
}


NB_MODULE(pypopsift, m) {

    bind_config(m);
}
