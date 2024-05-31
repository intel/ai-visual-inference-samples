#include <pybind11/pybind11.h>
#include "visual_ai/memory_format.hpp"

namespace py = pybind11;

/*
 * This is the entrypoint for the pybind11
 * Will dispatch all the pybind11 calls to the respective classes
 */

// Using forward function declaration to avoid importing the whole library

// Forward function declaration @xpu_decoder.cpp
void pybind11_submodule_videoreader(py::module_&);
// Forward function declaration @xpu_encoder.cpp
void pybind11_submodule_videowriter(py::module_&);
// Forward function declaration @frame_transform.cpp
void pybind11_submodule_transform(py::module_&);
// Forward function declaration @overlay_bind.cpp
void pybind11_init_ex_overlay(py::module&);
// Forward function declaration @utils_bind.cpp
void pybind11_init_ex_utils(py::module&);

PYBIND11_MODULE(libvisual_ai, m) {
    py::enum_<MemoryFormat>(m, "XpuMemoryFormat")
        .value("torch_contiguous_format", MemoryFormat::pt_planar_rgbp)
        .value("torch_channels_last", MemoryFormat::pt_packed_rgba)
        .value("openvino_planar", MemoryFormat::ov_planar_nv12)
        .value("system_rgbp", MemoryFormat::system_rgbp);

    // Decoder
    pybind11_submodule_videoreader(m);

    // Encoder
    pybind11_submodule_videowriter(m);

    // Transform
    pybind11_submodule_transform(m);

    // Overlay
    pybind11_init_ex_overlay(m);

    // Utils
    pybind11_init_ex_utils(m);
}
