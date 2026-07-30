#pragma once

#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define DEF_RELEASE_GIL(name, fn, doc) \
    m.def(name, fn, doc, py::call_guard<py::gil_scoped_release>())
