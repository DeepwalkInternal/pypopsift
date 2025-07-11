#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <popsift/popsift.h>
#include <popsift/sift_conf.h>
#include <popsift/features.h>

namespace nb = nanobind;

NB_MODULE(pypopsift, m) {
    m.doc() = "Python bindings for the PopSift library, a real-time SIFT implementation in CUDA.";

    // --- Enumerations for PopSift ---
    nb::enum_<PopSift::ImageMode>(m, "ImageMode", "Image modes for PopSift.")
        .value("ByteImages", PopSift::ImageMode::ByteImages, "8-bit grayscale images.")
        .value("FloatImages", PopSift::ImageMode::FloatImages, "32-bit float images.");

    nb::enum_<PopSift::AllocTest>(m, "AllocTest", "Results for texture allocation tests.")
        .value("Ok", PopSift::AllocTest::Ok, "Allocation will succeed.")
        .value("ImageExceedsLinearTextureLimit", PopSift::AllocTest::ImageExceedsLinearTextureLimit)
        .value("ImageExceedsLayeredSurfaceLimit", PopSift::AllocTest::ImageExceedsLayeredSurfaceLimit);

    // --- ADDED THIS ENUM BINDING ---
    nb::enum_<popsift::Config::ProcessingMode>(m, "ProcessingMode", "Processing mode for PopSift")
        .value("ExtractingMode", popsift::Config::ProcessingMode::ExtractingMode)
        .value("MatchingMode", popsift::Config::ProcessingMode::MatchingMode);


    // --- Enumerations for Config ---
    nb::enum_<popsift::Config::GaussMode>(m, "GaussMode").value("VLFeat_Compute", popsift::Config::GaussMode::VLFeat_Compute).value("VLFeat_Relative", popsift::Config::GaussMode::VLFeat_Relative).value("OpenCV_Compute", popsift::Config::GaussMode::OpenCV_Compute);
    nb::enum_<popsift::Config::SiftMode>(m, "SiftMode").value("PopSift", popsift::Config::SiftMode::PopSift).value("OpenCV", popsift::Config::SiftMode::OpenCV).value("VLFeat", popsift::Config::SiftMode::VLFeat);
    nb::enum_<popsift::Config::DescMode>(m, "DescMode").value("Loop", popsift::Config::DescMode::Loop).value("ILoop", popsift::Config::DescMode::ILoop).value("Grid", popsift::Config::DescMode::Grid).value("IGrid", popsift::Config::DescMode::IGrid).value("NoTile", popsift::Config::DescMode::NoTile);
    nb::enum_<popsift::Config::NormMode>(m, "NormMode").value("RootSift", popsift::Config::NormMode::RootSift).value("Classic", popsift::Config::NormMode::Classic);

    // --- Result Data Structures ---
    nb::class_<popsift::Descriptor>(m, "Descriptor", "SIFT feature descriptor.")
        .def_ro("features", &popsift::Descriptor::features);

    nb::class_<popsift::Feature>(m, "Feature", "Represents a single SIFT feature point.")
        .def_ro("xpos", &popsift::Feature::xpos)
        .def_ro("ypos", &popsift::Feature::ypos)
        .def_ro("sigma", &popsift::Feature::sigma)
        .def_ro("num_ori", &popsift::Feature::num_ori)
        .def_ro("orientation", &popsift::Feature::orientation)
        .def_ro("desc", &popsift::Feature::desc);
        
    nb::class_<popsift::FeaturesHost>(m, "FeaturesHost", "Container for SIFT features on the host.")
        .def("getFeatureCount", &popsift::FeaturesHost::getFeatureCount)
        .def("getDescriptorCount", &popsift::FeaturesHost::getDescriptorCount)
        .def("getFeatures", &popsift::FeaturesHost::getFeatures, nb::rv_policy::reference_internal)
        .def("getDescriptors", &popsift::FeaturesHost::getDescriptors, nb::rv_policy::reference_internal);
        
    nb::class_<popsift::FeaturesDev>(m, "FeaturesDev", "Container for SIFT features on the device.")
        .def("getFeatureCount", &popsift::FeaturesDev::getFeatureCount)
        .def("getDescriptorCount", &popsift::FeaturesDev::getDescriptorCount)
        .def("match", &popsift::FeaturesDev::match, "Matches features against another set of features on the GPU.");

    // --- SiftJob Class ---
    nb::class_<SiftJob>(m, "SiftJob", "Represents an asynchronous feature extraction task.")
        .def("getHost", &SiftJob::getHost, nb::rv_policy::take_ownership, "Waits for the job and returns features on the host.")
        .def("getDev", &SiftJob::getDev, nb::rv_policy::take_ownership, "Waits for the job and returns features on the device.");

    // --- Config Class ---
    nb::class_<popsift::Config>(m, "Config", "Configuration for the SIFT algorithm.")
        .def(nb::init<>())
        .def_rw("octaves", &popsift::Config::octaves)
        .def_rw("levels", &popsift::Config::levels)
        .def_rw("sigma", &popsift::Config::sigma)
        .def("setEdgeLimit", &popsift::Config::setEdgeLimit)
        .def("setThreshold", &popsift::Config::setThreshold)
        .def("setInitialBlur", &popsift::Config::setInitialBlur)
        .def("setGaussMode", static_cast<void (popsift::Config::*)(const std::string&)>(&popsift::Config::setGaussMode))
        .def("setMode", &popsift::Config::setMode)
        .def("setDescMode", static_cast<void (popsift::Config::*)(const std::string&)>(&popsift::Config::setDescMode))
        .def("setNormMode", static_cast<void (popsift::Config::*)(const std::string&)>(&popsift::Config::setNormMode))
        .def("setFilterMaxExtrema", &popsift::Config::setFilterMaxExtrema)
        .def("setFilterGridSize", &popsift::Config::setFilterGridSize)
        .def("setFilterSorting", static_cast<void (popsift::Config::*)(const std::string&)>(&popsift::Config::setFilterSorting));


    // --- Main PopSift Class ---
    nb::class_<PopSift>(m, "PopSift", "The main class for SIFT feature extraction.")
        .def(nb::init<popsift::Config, popsift::Config::ProcessingMode, PopSift::ImageMode, int>(), 
             nb::arg("config"),
             nb::arg("mode") = popsift::Config::ProcessingMode::ExtractingMode,
             nb::arg("imode") = PopSift::ByteImages, 
             nb::arg("device") = 0)
        .def("configure", &PopSift::configure, nb::arg("config"), nb::arg("force") = false)
        .def("uninit", static_cast<void (PopSift::*)()>(&PopSift::uninit))
        .def("testTextureFit", &PopSift::testTextureFit, nb::arg("width"), nb::arg("height"))
        .def("testTextureFitErrorString", &PopSift::testTextureFitErrorString, nb::arg("err"), nb::arg("width"), nb::arg("height"))
        .def("enqueue", [](PopSift& self, nb::ndarray<const uint8_t, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu> image) {
            return self.enqueue(image.shape(1), image.shape(0), image.data());
        }, nb::keep_alive<0, 1>())

        .def("enqueue", [](PopSift& self, nb::ndarray<const float, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu> image) {
            return self.enqueue(image.shape(1), image.shape(0), image.data());
        }, nb::keep_alive<0, 1>());
}
