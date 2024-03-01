#include <pybind11/pybind11.h>
#include "vaapi_utils.hpp"

namespace py = pybind11;

void pybind11_init_ex_utils(py::module& m) {
    m.def("dump_va_surface", &dump_va_surface, "Utility function that dumps VASurface into a file");
}