#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include "vaapi_overlay.hpp"

namespace py = pybind11;

void pybind11_init_ex_overlay(py::module& m) {
    py::class_<VaApiOverlay>(m, "VaApiOverlay")
        .def(py::init<VaDpyWrapper>())
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
            return OverlayBox{.x1 = tup[0].cast<int>(),
                              .y1 = tup[1].cast<int>(),
                              .x2 = tup[2].cast<int>(),
                              .y2 = tup[3].cast<int>()};
        }))
        .def(py::init([](int x1, int y1, int x2, int y2) {
            return OverlayBox{.x1 = x1, .y1 = y1, .x2 = x2, .y2 = y2};
        }))
        .def_readwrite("x1", &OverlayBox::x1)
        .def_readwrite("y1", &OverlayBox::y1)
        .def_readwrite("x2", &OverlayBox::x2)
        .def_readwrite("y2", &OverlayBox::y2)
        .def("astuple", [](const OverlayBox& self) {
            return py::make_tuple(self.x1, self.y1, self.x2, self.y2);
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
            oss << "<libvisual_ai.OverlayText x=" << ot.x << ", y=" << ot.y << ", text=" << ot.text
                << ">\n";
            return oss.str();
        });

    py::implicitly_convertible<py::tuple, OverlayBox>();
    py::implicitly_convertible<py::tuple, OverlayText>();
}
