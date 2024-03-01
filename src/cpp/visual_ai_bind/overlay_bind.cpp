#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include "vaapi_overlay.hpp"

namespace py = pybind11;

void pybind11_init_ex_overlay(py::module& m) {
    py::class_<VaApiOverlay>(m, "VaApiOverlay")
        .def(py::init<VADisplay>())
        .def("draw", &VaApiOverlay::draw, py::return_value_policy::take_ownership)
        .def("set_sync", &VaApiOverlay::set_sync)
        .def("stats", [](const VaApiOverlay& self) {
            py::dict d;
            const auto s = self.stats();
            d["last_total_us"] = s.last_total.count();
            d["last_mask_prepare_us"] = s.last_mask_prepare.count();
            d["last_mask_to_surface_us"] = s.last_mask_to_surface.count();
            d["last_blend_us"] = s.last_blend.count();
            d["is_sync"] = self.get_sync();
            return d;
        });

    py::class_<OverlayBox>(m, "OverlayBox")
        .def(pybind11::init<>())
        .def(py::init([](const py::tuple& tup) {
            if (py::len(tup) != 4) {
                throw py::cast_error("Invalid size: 4 elements are expected");
            }
            return OverlayBox{.x = tup[0].cast<int>(),
                              .y = tup[1].cast<int>(),
                              .width = tup[2].cast<int>(),
                              .height = tup[3].cast<int>()};
        }))
        .def(py::init([](int x, int y, int width, int height) {
            return OverlayBox{.x = x, .y = y, .width = width, .height = height};
        }))
        .def_readwrite("x", &OverlayBox::x)
        .def_readwrite("y", &OverlayBox::y)
        .def_readwrite("width", &OverlayBox::width)
        .def_readwrite("height", &OverlayBox::height)
        .def("astuple", [](const OverlayBox& self) {
            return py::make_tuple(self.x, self.y, self.width, self.height);
        });

    py::class_<OverlayText>(m, "OverlayText")
        .def(py::init([](int x, int y, std::string text) {
            return OverlayText{.x = x, .y = y, .text = std::move(text)};
        }))
        .def(py::init([](const py::tuple& tup) {
            if (py::len(tup) != 3) {
                throw py::cast_error("Invalid size: 3 elements are expected");
            }
            return OverlayText{.x = tup[0].cast<int>(),
                               .y = tup[1].cast<int>(),
                               .text = tup[2].cast<std::string>()};
        }))
        .def_readwrite("x", &OverlayText::x)
        .def_readwrite("y", &OverlayText::y)
        .def_readwrite("text", &OverlayText::text)
        .def("__repr__", [](const OverlayText& ot) {
            std::ostringstream oss;
            oss << "<libvideoreader.OverlayText x=" << ot.x << ", y=" << ot.y
                << ", text=" << ot.text << ">\n";
            return oss.str();
        });

    py::implicitly_convertible<py::tuple, OverlayBox>();
    py::implicitly_convertible<py::tuple, OverlayText>();
}