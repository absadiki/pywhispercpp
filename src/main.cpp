/**
 ********************************************************************************
 * @file    main.cpp
 * @author  [absadiki](https://github.com/absadiki)
 * @date    2023
 * @brief   Python bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) using Pybind11
 *
 * @par
 * COPYRIGHT NOTICE: (c) 2023.  All rights reserved.
 ********************************************************************************
 */

#include "whisper_bindings.cpp"

PYBIND11_MODULE(_pywhispercpp, m) {
    m.doc() = R"pbdoc(
        Pywhispercpp: Python binding to whisper.cpp
        -----------------------

        .. currentmodule:: _whispercpp

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    whisper_bindings(m);
}
