/**
 * Declares Parakeet pybind11 registration for _pywhispercpp.
 */
#pragma once

#include <pybind11/pybind11.h>

void init_parakeet_bindings(pybind11::module_ & m);
