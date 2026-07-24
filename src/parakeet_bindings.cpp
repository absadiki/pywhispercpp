/**
 * Python bindings for whisper.cpp Parakeet API (parakeet.h).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "parakeet.h"

#include <cstring>

namespace py = pybind11;

// Opaque context wrapper (parakeet_context is incomplete in the public header).
struct parakeet_context_wrapper {
    parakeet_context * ptr;
};

static py::object parakeet_init_from_file_with_params_wrapper(
        const char * path_model,
        struct parakeet_context_params cparams) {
    // Release the GIL only for the native load. py::cast / py::none need the GIL.
    struct parakeet_context * ctx;
    {
        py::gil_scoped_release release;
        ctx = parakeet_init_from_file_with_params(path_model, cparams);
    }
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
    return py::bytes(c_array, strlen(c_array));
}

void init_parakeet_bindings(py::module_ & m) {
    m.attr("PARAKEET_SAMPLE_RATE") = PARAKEET_SAMPLE_RATE;

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

    m.def("parakeet_version", &parakeet_version);
    m.def("parakeet_context_default_params", &parakeet_context_default_params);
    m.def("parakeet_full_default_params", &parakeet_full_default_params, py::arg("strategy"));
    // No call_guard gil_scoped_release: this binding returns py::object via py::cast.
    m.def("parakeet_init_from_file_with_params",
          &parakeet_init_from_file_with_params_wrapper,
          py::arg("path_model"), py::arg("params"));
    m.def("parakeet_free", &parakeet_free_wrapper);
    m.def("parakeet_full", &parakeet_full_wrapper,
          py::arg("ctx"), py::arg("params"), py::arg("samples"), py::arg("n_samples"));
    m.def("parakeet_full_n_segments", &parakeet_full_n_segments_wrapper);
    m.def("parakeet_full_get_segment_t0", &parakeet_full_get_segment_t0_wrapper);
    m.def("parakeet_full_get_segment_t1", &parakeet_full_get_segment_t1_wrapper);
    m.def("parakeet_full_get_segment_text", &parakeet_full_get_segment_text_wrapper);
}
