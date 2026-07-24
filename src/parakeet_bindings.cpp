/**
 * Python bindings for whisper.cpp Parakeet API (parakeet.h).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "parakeet.h"

#include <cstring>
#include <string>

namespace py = pybind11;

// Opaque context wrapper (parakeet_context is incomplete in the public header).
struct parakeet_context_wrapper {
    parakeet_context * ptr;
};

static py::object parakeet_init_from_file_with_params_wrapper(
        const char * path_model,
        struct parakeet_context_params cparams) {
    struct parakeet_context * ctx = parakeet_init_from_file_with_params(path_model, cparams);
    if (!ctx) {
        return py::none();
    }
    struct parakeet_context_wrapper w;
    w.ptr = ctx;
    return py::cast(w);
}

static void parakeet_free_wrapper(py::object ctx_obj) {
    if (ctx_obj.is_none()) {
        return;
    }
    auto * ctx_w = ctx_obj.cast<parakeet_context_wrapper *>();
    if (ctx_w && ctx_w->ptr) {
        parakeet_free(ctx_w->ptr);
        ctx_w->ptr = nullptr;
    }
}

static int parakeet_full_wrapper(
        struct parakeet_context_wrapper * ctx_w,
        struct parakeet_full_params params,
        py::array_t<float> samples,
        int n_samples) {
    if (!ctx_w || !ctx_w->ptr) {
        return -1;
    }
    py::buffer_info buf = samples.request();
    float * samples_ptr = static_cast<float *>(buf.ptr);

    py::gil_scoped_release release;
    return parakeet_full(ctx_w->ptr, params, samples_ptr, n_samples);
}

static int parakeet_full_n_segments_wrapper(struct parakeet_context_wrapper * ctx_w) {
    if (!ctx_w || !ctx_w->ptr) {
        return 0;
    }
    py::gil_scoped_release release;
    return parakeet_full_n_segments(ctx_w->ptr);
}

static int64_t parakeet_full_get_segment_t0_wrapper(
        struct parakeet_context_wrapper * ctx_w, int i_segment) {
    return parakeet_full_get_segment_t0(ctx_w->ptr, i_segment);
}

static int64_t parakeet_full_get_segment_t1_wrapper(
        struct parakeet_context_wrapper * ctx_w, int i_segment) {
    return parakeet_full_get_segment_t1(ctx_w->ptr, i_segment);
}

static py::bytes parakeet_full_get_segment_text_wrapper(
        struct parakeet_context_wrapper * ctx_w, int i_segment) {
    const char * c_array = parakeet_full_get_segment_text(ctx_w->ptr, i_segment);
    if (!c_array) {
        return py::bytes("", 0);
    }
    size_t length = strlen(c_array);
    return py::bytes(c_array, length);
}

void init_parakeet_bindings(py::module_ & m) {
    m.attr("PARAKEET_SAMPLE_RATE") = PARAKEET_SAMPLE_RATE;
    m.attr("PARAKEET_HOP_LENGTH") = PARAKEET_HOP_LENGTH;

    py::enum_<parakeet_sampling_strategy>(m, "parakeet_sampling_strategy")
        .value("PARAKEET_SAMPLING_GREEDY", parakeet_sampling_strategy::PARAKEET_SAMPLING_GREEDY)
        .export_values();

    py::class_<parakeet_context_wrapper>(m, "parakeet_context");

    py::class_<parakeet_context_params>(m, "parakeet_context_params")
        .def(py::init<>())
        .def_readwrite("use_gpu", &parakeet_context_params::use_gpu)
        .def_readwrite("gpu_device", &parakeet_context_params::gpu_device);

    py::class_<parakeet_full_params>(m, "parakeet_full_params")
        .def(py::init<>())
        .def_readwrite("strategy", &parakeet_full_params::strategy)
        .def_readwrite("n_threads", &parakeet_full_params::n_threads)
        .def_readwrite("offset_ms", &parakeet_full_params::offset_ms)
        .def_readwrite("duration_ms", &parakeet_full_params::duration_ms)
        .def_readwrite("no_context", &parakeet_full_params::no_context)
        .def_readwrite("audio_ctx", &parakeet_full_params::audio_ctx);

    m.def("parakeet_version", &parakeet_version,
          "Return the Parakeet library version string.");

    m.def("parakeet_context_default_params", &parakeet_context_default_params,
          "Return default Parakeet context parameters.");

    m.def("parakeet_full_default_params", &parakeet_full_default_params,
          "Return default Parakeet full-params for the given sampling strategy.",
          py::arg("strategy"));

    m.def("parakeet_init_from_file_with_params",
          &parakeet_init_from_file_with_params_wrapper,
          "Load a Parakeet GGUF model from path. Returns a context (ptr may be null on failure).",
          py::arg("path_model"), py::arg("params"),
          py::call_guard<py::gil_scoped_release>());

    m.def("parakeet_free", &parakeet_free_wrapper,
          "Free memory allocated for a Parakeet context.");

    m.def("parakeet_full", &parakeet_full_wrapper,
          "Run full Parakeet inference on float32 PCM samples (16 kHz mono).",
          py::arg("ctx"), py::arg("params"), py::arg("samples"), py::arg("n_samples"));

    m.def("parakeet_full_n_segments", &parakeet_full_n_segments_wrapper,
          "Number of text segments produced by the last parakeet_full() call.");

    m.def("parakeet_full_get_segment_t0", &parakeet_full_get_segment_t0_wrapper,
          "Start time of segment i (centiseconds, same convention as whisper).");

    m.def("parakeet_full_get_segment_t1", &parakeet_full_get_segment_t1_wrapper,
          "End time of segment i (centiseconds, same convention as whisper).");

    m.def("parakeet_full_get_segment_text", &parakeet_full_get_segment_text_wrapper,
          "Text of segment i as bytes (decode as UTF-8).");

    m.def("parakeet_print_system_info", &parakeet_print_system_info,
          "Print system / backend information for Parakeet.");
}
