#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "parakeet.h"
#include "bindings_utils.h"


namespace py = pybind11;
using namespace pybind11::literals;

inline bool has_parakeet_python_user_data(const py::object & obj) {
    return obj.ptr() != nullptr && obj.ptr() != Py_None;
}

py::object py_parakeet_log_callback;


// parakeet context wrapper, to solve the incomplete type issue
struct parakeet_context_wrapper {
    parakeet_context* ptr;
};

struct parakeet_context_wrapper parakeet_init_from_file_with_params_wrapper(
        const char * path_model,
        struct parakeet_context_params cparams){
    struct parakeet_context * ctx = parakeet_init_from_file_with_params(path_model, cparams);
    struct parakeet_context_wrapper ctw_w;
    ctw_w.ptr = ctx;
    return ctw_w;
}

struct parakeet_context_wrapper parakeet_init_from_buffer_with_params_wrapper(
        void * buffer,
        size_t buffer_size,
        struct parakeet_context_params cparams){
    struct parakeet_context * ctx = parakeet_init_from_buffer_with_params(buffer, buffer_size, cparams);
    struct parakeet_context_wrapper ctw_w;
    ctw_w.ptr = ctx;
    return ctw_w;
}

void parakeet_free_wrapper(struct parakeet_context_wrapper * ctx_w){
    parakeet_free(ctx_w->ptr);
};

int parakeet_pcm_to_mel_wrapper(
        struct parakeet_context_wrapper * ctx,
        py::array_t<float> samples,
        int   n_samples,
        int   n_threads){
    py::buffer_info buf = samples.request();
    float *samples_ptr = static_cast<float *>(buf.ptr);
    py::gil_scoped_release release;
    return parakeet_pcm_to_mel(ctx->ptr, samples_ptr, n_samples, n_threads);
};

int parakeet_set_mel_wrapper(
        struct parakeet_context_wrapper * ctx,
        py::array_t<float> data,
        int   n_len,
        int   n_mel){
    py::buffer_info buf = data.request();
    float *data_ptr = static_cast<float *>(buf.ptr);
    return parakeet_set_mel(ctx->ptr, data_ptr, n_len, n_mel);
};

int parakeet_encode_wrapper(
        struct parakeet_context_wrapper * ctx,
        int   offset,
        int   n_threads){
    py::gil_scoped_release release;
    return parakeet_encode(ctx->ptr, offset, n_threads);
}

int parakeet_tokenize_wrapper(
        struct parakeet_context_wrapper * ctx,
        const char * text,
        parakeet_token * tokens,
        int   n_max_tokens){
    return parakeet_tokenize(ctx->ptr, text, tokens, n_max_tokens);
};

int parakeet_n_len_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_n_len(ctx_w->ptr);
};

int parakeet_n_vocab_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_n_vocab(ctx_w->ptr);
};

int parakeet_n_audio_ctx_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_n_audio_ctx(ctx_w->ptr);
}

float * parakeet_get_logits_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_get_logits(ctx_w->ptr);
};

const char * parakeet_token_to_str_wrapper(struct parakeet_context_wrapper * ctx_w, parakeet_token token){
    return parakeet_token_to_str(ctx_w->ptr, token);
};

py::bytes parakeet_token_to_bytes_wrapper(struct parakeet_context_wrapper * ctx_w, parakeet_token token){
    const char* str = parakeet_token_to_str(ctx_w->ptr, token);
    size_t l = strlen(str);
    return py::bytes(str, l);
}

parakeet_token parakeet_token_blank_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_token_blank(ctx_w->ptr);
}

parakeet_token parakeet_token_unk_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_token_unk(ctx_w->ptr);
}

parakeet_token parakeet_token_bos_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_token_bos(ctx_w->ptr);
}

void parakeet_print_timings_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_print_timings(ctx_w->ptr);
}

void parakeet_reset_timings_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_reset_timings(ctx_w->ptr);
}

int parakeet_full_wrapper(
        struct parakeet_context_wrapper * ctx_w,
        struct parakeet_full_params   params,
        py::array_t<float> samples,
        int   n_samples){
    py::buffer_info buf = samples.request();
    float *samples_ptr = static_cast<float *>(buf.ptr);

    py::gil_scoped_release release;
    return parakeet_full(ctx_w->ptr, params, samples_ptr, n_samples);
}

int parakeet_full_n_segments_wrapper(struct parakeet_context_wrapper * ctx){
    return parakeet_full_n_segments(ctx->ptr);
}

int64_t parakeet_full_get_segment_t0_wrapper(struct parakeet_context_wrapper * ctx, int i_segment){
    return parakeet_full_get_segment_t0(ctx->ptr, i_segment);
}

int64_t parakeet_full_get_segment_t1_wrapper(struct parakeet_context_wrapper * ctx, int i_segment){
    return parakeet_full_get_segment_t1(ctx->ptr, i_segment);
}

const py::bytes parakeet_full_get_segment_text_wrapper(struct parakeet_context_wrapper * ctx, int i_segment){
    const char * c_array = parakeet_full_get_segment_text(ctx->ptr, i_segment);
    size_t length = strlen(c_array);
    return py::bytes(c_array, length);
};

int parakeet_full_n_tokens_wrapper(struct parakeet_context_wrapper * ctx, int i_segment){
    return parakeet_full_n_tokens(ctx->ptr, i_segment);
}

const char * parakeet_full_get_token_text_wrapper(struct parakeet_context_wrapper * ctx, int i_segment, int i_token){
    return parakeet_full_get_token_text(ctx->ptr, i_segment, i_token);
}

parakeet_token parakeet_full_get_token_id_wrapper(struct parakeet_context_wrapper * ctx, int i_segment, int i_token){
    return parakeet_full_get_token_id(ctx->ptr, i_segment, i_token);
}

parakeet_token_data parakeet_full_get_token_data_wrapper(struct parakeet_context_wrapper * ctx, int i_segment, int i_token){
    return parakeet_full_get_token_data(ctx->ptr, i_segment, i_token);
}

float parakeet_full_get_token_p_wrapper(struct parakeet_context_wrapper * ctx, int i_segment, int i_token){
    return parakeet_full_get_token_p(ctx->ptr, i_segment, i_token);
}

int parakeet_model_n_vocab_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_model_n_vocab(ctx_w->ptr);
}

int parakeet_model_n_audio_ctx_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_model_n_audio_ctx(ctx_w->ptr);
}

int parakeet_model_n_audio_state_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_model_n_audio_state(ctx_w->ptr);
}

int parakeet_model_n_audio_head_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_model_n_audio_head(ctx_w->ptr);
}

int parakeet_model_n_audio_layer_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_model_n_audio_layer(ctx_w->ptr);
}

int parakeet_model_n_mels_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_model_n_mels(ctx_w->ptr);
}

int parakeet_model_ftype_wrapper(struct parakeet_context_wrapper * ctx_w){
    return parakeet_model_ftype(ctx_w->ptr);
}

// forward declarations for callbacks
bool _parakeet_abort_callback(void * user_data);
void _parakeet_new_segment_callback(struct parakeet_context * ctx, struct parakeet_state * state, int n_new, void * user_data);
bool _parakeet_encoder_begin_callback(struct parakeet_context * ctx, struct parakeet_state * state, void * user_data);

struct ParakeetFullParamsWrapper : public parakeet_full_params {
    ParakeetFullParamsWrapper(const parakeet_full_params& params = parakeet_full_params())
        : parakeet_full_params(params),
          py_new_segment_callback_user_data(py::none()),
          py_encoder_begin_callback_user_data(py::none()),
          py_progress_callback_user_data(py::none()),
          py_abort_callback_user_data(py::none())
    {
        new_segment_callback_user_data = this;
        encoder_begin_callback_user_data = this;
        abort_callback_user_data = this;
        progress_callback_user_data = this;
    }

    ParakeetFullParamsWrapper(const ParakeetFullParamsWrapper& other)
        : parakeet_full_params(static_cast<parakeet_full_params>(other)),
          py_new_segment_callback(other.py_new_segment_callback),
          py_new_segment_callback_user_data(other.py_new_segment_callback_user_data),
          py_encoder_begin_callback(other.py_encoder_begin_callback),
          py_encoder_begin_callback_user_data(other.py_encoder_begin_callback_user_data),
          py_progress_callback(other.py_progress_callback),
          py_progress_callback_user_data(other.py_progress_callback_user_data),
          py_abort_callback(other.py_abort_callback),
          py_abort_callback_user_data(other.py_abort_callback_user_data)
    {
        new_segment_callback_user_data = this;
        encoder_begin_callback_user_data = this;
        abort_callback_user_data = this;
        progress_callback_user_data = this;
    }

public:
    py::function py_new_segment_callback;
    py::object py_new_segment_callback_user_data;
    py::function py_encoder_begin_callback;
    py::object py_encoder_begin_callback_user_data;
    py::function py_progress_callback;
    py::object py_progress_callback_user_data;
    py::function py_abort_callback;
    py::object py_abort_callback_user_data;

    py::object get_new_segment_callback_user_data() const {
        return py_new_segment_callback_user_data;
    }
    void set_new_segment_callback_user_data(py::object user_data) {
        py_new_segment_callback_user_data = std::move(user_data);
        new_segment_callback_user_data = this;
    }
    void set_new_segment_callback(py::function callback) {
        py_new_segment_callback = std::move(callback);
        new_segment_callback_user_data = this;
        new_segment_callback = _parakeet_new_segment_callback;
    }
    void clear_new_segment_callback() {
        py_new_segment_callback = py::function();
        new_segment_callback = nullptr;
        new_segment_callback_user_data = this;
    }

    py::object get_encoder_begin_callback_user_data() const {
        return py_encoder_begin_callback_user_data;
    }
    void set_encoder_begin_callback_user_data(py::object user_data) {
        py_encoder_begin_callback_user_data = std::move(user_data);
        encoder_begin_callback_user_data = this;
    }
    void set_encoder_begin_callback(py::function callback) {
        py_encoder_begin_callback = std::move(callback);
        encoder_begin_callback_user_data = this;
        encoder_begin_callback = _parakeet_encoder_begin_callback;
    }
    void clear_encoder_begin_callback() {
        py_encoder_begin_callback = py::function();
        encoder_begin_callback = nullptr;
        encoder_begin_callback_user_data = this;
    }

    py::object get_progress_callback_user_data() const {
        return py_progress_callback_user_data;
    }
    void set_progress_callback_user_data(py::object user_data) {
        py_progress_callback_user_data = std::move(user_data);
        progress_callback_user_data = this;
    }
    void set_progress_callback(py::function callback) {
        py_progress_callback = std::move(callback);
        progress_callback_user_data = this;
        progress_callback = [](struct parakeet_context* ctx, struct parakeet_state* state, int progress, void* user_data) {
            (void) ctx;
            (void) state;
            auto* self = static_cast<ParakeetFullParamsWrapper*>(user_data);
            if (self && self->py_progress_callback) {
                py::gil_scoped_acquire gil;
                if (!has_parakeet_python_user_data(self->py_progress_callback_user_data)) {
                    self->py_progress_callback(progress);
                } else {
                    self->py_progress_callback(progress, self->py_progress_callback_user_data);
                }
            }
        };
    }
    void clear_progress_callback() {
        py_progress_callback = py::function();
        progress_callback = nullptr;
        progress_callback_user_data = this;
    }

    py::object get_abort_callback_user_data() const {
        return py_abort_callback_user_data;
    }
    void set_abort_callback_user_data(py::object user_data) {
        py_abort_callback_user_data = std::move(user_data);
        abort_callback_user_data = this;
    }
    void set_abort_callback(py::function callback) {
        py_abort_callback = std::move(callback);
        abort_callback_user_data = this;
        abort_callback = _parakeet_abort_callback;
    }
    void clear_abort_callback() {
        py_abort_callback = py::function();
        abort_callback = nullptr;
        abort_callback_user_data = this;
    }
};

ParakeetFullParamsWrapper parakeet_full_default_params_wrapper(enum parakeet_sampling_strategy strategy) {
    return ParakeetFullParamsWrapper(parakeet_full_default_params(strategy));
}

// callbacks mechanism

void _parakeet_new_segment_callback(struct parakeet_context * ctx, struct parakeet_state * state, int n_new, void * user_data){
    (void) state;
    struct parakeet_context_wrapper ctx_w;
    ctx_w.ptr = ctx;
    auto * params = static_cast<ParakeetFullParamsWrapper *>(user_data);
    if (!params || !params->py_new_segment_callback) {
        return;
    }

    py::gil_scoped_acquire gil;
    py::function callback = params->py_new_segment_callback;
    if (!has_parakeet_python_user_data(params->py_new_segment_callback_user_data)) {
        callback(ctx_w, n_new);
    } else {
        callback(ctx_w, n_new, params->py_new_segment_callback_user_data);
    }
};

void parakeet_assign_new_segment_callback(parakeet_full_params *params_base, py::object callback){
    auto * params = static_cast<ParakeetFullParamsWrapper *>(params_base);
    if (callback.is_none()) {
        params->clear_new_segment_callback();
        return;
    }

    params->set_new_segment_callback(callback.cast<py::function>());
}

void parakeet_clear_new_segment_callback(parakeet_full_params *params_base) {
    auto * params = static_cast<ParakeetFullParamsWrapper *>(params_base);
    params->clear_new_segment_callback();
};

bool _parakeet_encoder_begin_callback(struct parakeet_context * ctx, struct parakeet_state * state, void * user_data){
    (void) state;
    struct parakeet_context_wrapper ctx_w;
    ctx_w.ptr = ctx;
    auto * params = static_cast<ParakeetFullParamsWrapper *>(user_data);
    if (!params || !params->py_encoder_begin_callback) {
        return false;
    }

    py::gil_scoped_acquire gil;
    py::function callback = params->py_encoder_begin_callback;
    py::object result_py;
    if (!has_parakeet_python_user_data(params->py_encoder_begin_callback_user_data)) {
        result_py = callback(ctx_w);
    } else {
        result_py = callback(ctx_w, params->py_encoder_begin_callback_user_data);
    }
    bool res = result_py.cast<bool>();
    return res;
}

void parakeet_assign_encoder_begin_callback(parakeet_full_params *params_base, py::object callback){
    auto * params = static_cast<ParakeetFullParamsWrapper *>(params_base);
    if (callback.is_none()) {
        params->clear_encoder_begin_callback();
        return;
    }

    params->set_encoder_begin_callback(callback.cast<py::function>());
}

void parakeet_clear_encoder_begin_callback(parakeet_full_params *params_base) {
    auto * params = static_cast<ParakeetFullParamsWrapper *>(params_base);
    params->clear_encoder_begin_callback();
}

bool _parakeet_abort_callback(void * user_data) {
    auto * params = static_cast<ParakeetFullParamsWrapper *>(user_data);
    if (!params || !params->py_abort_callback) {
        return false;
    }

    py::gil_scoped_acquire gil;
    py::function callback = params->py_abort_callback;
    py::object result_py;
    if (!has_parakeet_python_user_data(params->py_abort_callback_user_data)) {
        result_py = callback();
    } else {
        result_py = callback(params->py_abort_callback_user_data);
    }
    return result_py.cast<bool>();
}

void parakeet_assign_abort_callback(parakeet_full_params *params_base, py::object callback){
    auto * params = static_cast<ParakeetFullParamsWrapper *>(params_base);
    if (callback.is_none()) {
        params->clear_abort_callback();
        return;
    }

    params->set_abort_callback(callback.cast<py::function>());
}

void parakeet_clear_abort_callback(parakeet_full_params *params_base) {
    auto * params = static_cast<ParakeetFullParamsWrapper *>(params_base);
    params->clear_abort_callback();
}

void parakeet_log_set_wrapper(py::object callback) {
    if (callback.is_none()) {
        py_parakeet_log_callback = py::none();
        parakeet_log_set(nullptr, nullptr);
        return;
    }

    py_parakeet_log_callback = callback.cast<py::function>();
    parakeet_log_set(
        [](enum ggml_log_level level, const char * text, void * user_data) {
            (void) user_data;
            py::gil_scoped_acquire gil;
            py::function log_callback = py_parakeet_log_callback.cast<py::function>();
            log_callback(py::int_(static_cast<int>(level)), py::str(text ? text : ""));
        },
        nullptr);
}


//////////

void parakeet_bindings(py::module_ & m) {

    m.attr("PARAKEET_SAMPLE_RATE") = PARAKEET_SAMPLE_RATE;
    m.attr("PARAKEET_HOP_LENGTH") = PARAKEET_HOP_LENGTH;

    py::class_<parakeet_context_wrapper>(m, "parakeet_context");
    py::class_<parakeet_context_params>(m, "parakeet_context_params")
        .def(py::init<>())
        .def_readwrite("use_gpu", &parakeet_context_params::use_gpu)
        .def_readwrite("gpu_device", &parakeet_context_params::gpu_device);

    py::class_<parakeet_token_data>(m, "parakeet_token_data")
        .def(py::init<>())
        .def_readwrite("id", &parakeet_token_data::id)
        .def_readwrite("duration_idx", &parakeet_token_data::duration_idx)
        .def_readwrite("duration_value", &parakeet_token_data::duration_value)
        .def_readwrite("frame_index", &parakeet_token_data::frame_index)
        .def_readwrite("p", &parakeet_token_data::p)
        .def_readwrite("plog", &parakeet_token_data::plog)
        .def_readwrite("t0", &parakeet_token_data::t0)
        .def_readwrite("t1", &parakeet_token_data::t1)
        .def_readwrite("is_word_start", &parakeet_token_data::is_word_start);

    m.def("parakeet_version", &parakeet_version, "Return the version of the parakeet library.");

    m.def("parakeet_context_default_params", &parakeet_context_default_params,
        "Return the default context parameters used during model initialization.");

    DEF_RELEASE_GIL("parakeet_init_from_file_with_params", &parakeet_init_from_file_with_params_wrapper,
        "Various functions for loading a ggml parakeet model.\n"
        "Allocate (almost) all memory needed for the model.\n"
        "Return NULL on failure");

    DEF_RELEASE_GIL("parakeet_init_from_buffer_with_params", &parakeet_init_from_buffer_with_params_wrapper,
        "Various functions for loading a ggml parakeet model.\n"
        "Allocate (almost) all memory needed for the model.\n"
        "Return NULL on failure");

    m.def("parakeet_free", &parakeet_free_wrapper, "Frees all memory allocated by the model.");

    m.def("parakeet_pcm_to_mel", &parakeet_pcm_to_mel_wrapper, "Convert RAW PCM audio to log mel spectrogram.\n"
                                                             "The resulting spectrogram is stored inside the provided parakeet context.\n"
                                                             "Returns 0 on success");

    m.def("parakeet_set_mel", &parakeet_set_mel_wrapper, "This can be used to set a custom log mel spectrogram inside the provided parakeet context.\n"
                                                        "Use this instead of parakeet_pcm_to_mel() if you want to provide your own log mel spectrogram.\n"
                                                        "n_mel must be 128\n"
                                                        "Returns 0 on success");

    m.def("parakeet_encode", &parakeet_encode_wrapper, "Run the Parakeet encoder on the log mel spectrogram stored inside the provided parakeet context.\n"
                                                    "Make sure to call parakeet_pcm_to_mel() or parakeet_set_mel() first.\n"
                                                    "offset can be used to specify the offset of the first frame in the spectrogram.\n"
                                                    "Returns 0 on success");

    m.def("parakeet_tokenize", &parakeet_tokenize_wrapper, "Convert the provided text into tokens.\n"
                                                        "The tokens pointer must be large enough to hold the resulting tokens.\n"
                                                        "Returns the number of tokens on success, no more than n_max_tokens\n"
                                                        "Returns a negative number on failure");

    m.def("parakeet_n_len", &parakeet_n_len_wrapper, "parakeet_n_len");
    m.def("parakeet_n_vocab", &parakeet_n_vocab_wrapper, "parakeet_n_vocab");
    m.def("parakeet_n_audio_ctx", &parakeet_n_audio_ctx_wrapper, "parakeet_n_audio_ctx");

    m.def("parakeet_get_logits", &parakeet_get_logits_wrapper, "Token logits obtained from the last call to parakeet_full()\n"
                                                             "The logits for the last token are stored in the last row\n"
                                                             "Rows: n_tokens\n"
                                                             "Cols: n_vocab");

    m.def("parakeet_token_to_str", &parakeet_token_to_str_wrapper, "parakeet_token_to_str");
    m.def("parakeet_token_to_bytes", &parakeet_token_to_bytes_wrapper, "parakeet_token_to_bytes");
    m.def("parakeet_token_blank", &parakeet_token_blank_wrapper, "parakeet_token_blank");
    m.def("parakeet_token_unk", &parakeet_token_unk_wrapper, "parakeet_token_unk");
    m.def("parakeet_token_bos", &parakeet_token_bos_wrapper, "parakeet_token_bos");

    m.def("parakeet_print_timings", &parakeet_print_timings_wrapper);
    m.def("parakeet_reset_timings", &parakeet_reset_timings_wrapper);

    m.def("parakeet_print_system_info", &parakeet_print_system_info);

    //////////////////////

    py::enum_<parakeet_sampling_strategy>(m, "parakeet_sampling_strategy")
        .value("PARAKEET_SAMPLING_GREEDY", parakeet_sampling_strategy::PARAKEET_SAMPLING_GREEDY)
        .export_values();

    py::class_<parakeet_full_params>(m, "__parakeet_full_params__internal")
        .def(py::init<>())
        .def("__repr__", [](const parakeet_full_params& self) {
            std::ostringstream oss;
            oss << "parakeet_full_params("
                << "strategy=" << self.strategy << ", "
                << "n_threads=" << self.n_threads << ", "
                << "offset_ms=" << self.offset_ms << ", "
                << "duration_ms=" << self.duration_ms << ", "
                << "no_context=" << (self.no_context ? "True" : "False") << ", "
                << "audio_ctx=" << self.audio_ctx << ", "
                << "new_segment_callback=" << (self.new_segment_callback ? "(function pointer)" : "None") << ", "
                << "progress_callback=" << (self.progress_callback ? "(function pointer)" : "None") << ", "
                << "encoder_begin_callback=" << (self.encoder_begin_callback ? "(function pointer)" : "None") << ", "
                << "abort_callback=" << (self.abort_callback ? "(function pointer)" : "None")
                << ")";
            return oss.str();
        });

    py::class_<ParakeetFullParamsWrapper, parakeet_full_params>(m, "parakeet_full_params")
        .def(py::init<>())
        .def_readwrite("strategy", &ParakeetFullParamsWrapper::strategy)
        .def_readwrite("n_threads", &ParakeetFullParamsWrapper::n_threads)
        .def_readwrite("offset_ms", &ParakeetFullParamsWrapper::offset_ms)
        .def_readwrite("duration_ms", &ParakeetFullParamsWrapper::duration_ms)
        .def_readwrite("no_context", &ParakeetFullParamsWrapper::no_context)
        .def_readwrite("audio_ctx", &ParakeetFullParamsWrapper::audio_ctx)
        .def_readwrite("progress_callback", &ParakeetFullParamsWrapper::py_progress_callback)
        .def("set_progress_callback",
             [](ParakeetFullParamsWrapper &self, py::object callback) {
                 if (callback.is_none()) {
                     self.clear_progress_callback();
                 } else {
                     self.set_progress_callback(callback.cast<py::function>());
                 }
             },
             py::arg("callback") = py::none(),
             "Assign a progress callback that receives progress updates.")
        .def("clear_progress_callback", &ParakeetFullParamsWrapper::clear_progress_callback,
             "Clear any previously assigned progress callback while preserving default progress behavior.")
        .def("set_new_segment_callback",
             [](ParakeetFullParamsWrapper &self, py::object callback) {
                 if (callback.is_none()) {
                     self.clear_new_segment_callback();
                 } else {
                     self.set_new_segment_callback(callback.cast<py::function>());
                 }
             },
             py::arg("callback") = py::none(),
             "Assign a new-segment callback.")
        .def("clear_new_segment_callback", &ParakeetFullParamsWrapper::clear_new_segment_callback,
             "Clear any previously assigned new-segment callback.")
        .def("set_encoder_begin_callback",
             [](ParakeetFullParamsWrapper &self, py::object callback) {
                 if (callback.is_none()) {
                     self.clear_encoder_begin_callback();
                 } else {
                     self.set_encoder_begin_callback(callback.cast<py::function>());
                 }
             },
             py::arg("callback") = py::none(),
             "Assign an encoder-begin callback.")
        .def("clear_encoder_begin_callback", &ParakeetFullParamsWrapper::clear_encoder_begin_callback,
             "Clear any previously assigned encoder-begin callback.")
        .def("set_abort_callback",
             [](ParakeetFullParamsWrapper &self, py::object callback) {
                 if (callback.is_none()) {
                     self.clear_abort_callback();
                 } else {
                     self.set_abort_callback(callback.cast<py::function>());
                 }
             },
             py::arg("callback") = py::none(),
             "Assign an abort callback that returns True to stop processing.")
        .def("clear_abort_callback", &ParakeetFullParamsWrapper::clear_abort_callback,
             "Clear any previously assigned abort callback.")
        .def_property("new_segment_callback_user_data",
            &ParakeetFullParamsWrapper::get_new_segment_callback_user_data,
            &ParakeetFullParamsWrapper::set_new_segment_callback_user_data)
        .def_property("progress_callback_user_data",
            &ParakeetFullParamsWrapper::get_progress_callback_user_data,
            &ParakeetFullParamsWrapper::set_progress_callback_user_data)
        .def_property("encoder_begin_callback_user_data",
            &ParakeetFullParamsWrapper::get_encoder_begin_callback_user_data,
            &ParakeetFullParamsWrapper::set_encoder_begin_callback_user_data)
        .def_property("abort_callback_user_data",
            &ParakeetFullParamsWrapper::get_abort_callback_user_data,
            &ParakeetFullParamsWrapper::set_abort_callback_user_data);

    py::implicitly_convertible<parakeet_full_params, ParakeetFullParamsWrapper>();

    m.def("parakeet_full_default_params", &parakeet_full_default_params_wrapper);

    m.def("parakeet_full", &parakeet_full_wrapper, "Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text\n"
                                                 "Uses the specified decoding strategy to obtain the text.\n");

    m.def("parakeet_full_n_segments", &parakeet_full_n_segments_wrapper, "Number of generated text segments.\n"
                                                                       "A segment can be a few words, a sentence, or even a paragraph.\n");

    m.def("parakeet_full_get_segment_t0", &parakeet_full_get_segment_t0_wrapper, "Get the start time of the specified segment");
    m.def("parakeet_full_get_segment_t1", &parakeet_full_get_segment_t1_wrapper, "Get the end time of the specified segment");
    m.def("parakeet_full_get_segment_text", &parakeet_full_get_segment_text_wrapper, "Get the text of the specified segment");
    m.def("parakeet_full_n_tokens", &parakeet_full_n_tokens_wrapper, "Get number of tokens in the specified segment.");

    m.def("parakeet_full_get_token_text", &parakeet_full_get_token_text_wrapper, "Get the token text of the specified token in the specified segment.");
    m.def("parakeet_full_get_token_id", &parakeet_full_get_token_id_wrapper, "Get the token id of the specified token in the specified segment.");

    m.def("parakeet_full_get_token_data", &parakeet_full_get_token_data_wrapper, "Get token data for the specified token in the specified segment.\n"
                                                                                "This contains probabilities, timestamps, etc.");

    m.def("parakeet_full_get_token_p", &parakeet_full_get_token_p_wrapper, "Get the probability of the specified token in the specified segment.");

    m.def("parakeet_model_n_vocab", &parakeet_model_n_vocab_wrapper, "Return the model vocabulary size.");
    m.def("parakeet_model_n_audio_ctx", &parakeet_model_n_audio_ctx_wrapper, "Return the audio context size baked into the model.");
    m.def("parakeet_model_n_audio_state", &parakeet_model_n_audio_state_wrapper, "Return the number of audio state units in the model.");
    m.def("parakeet_model_n_audio_head", &parakeet_model_n_audio_head_wrapper, "Return the number of audio attention heads in the model.");
    m.def("parakeet_model_n_audio_layer", &parakeet_model_n_audio_layer_wrapper, "Return the number of audio layers in the model.");
    m.def("parakeet_model_n_mels", &parakeet_model_n_mels_wrapper, "Return the number of mel bins used by the model.");
    m.def("parakeet_model_ftype", &parakeet_model_ftype_wrapper, "Return the model file type identifier.");

    ////////////////////////////////////////////////////////////////////////////
    // Helper mechanism to set callbacks from python

    m.def("parakeet_assign_new_segment_callback",
        [](parakeet_full_params * params, py::object callback) {
            parakeet_assign_new_segment_callback(params, callback);
        },
        "Assign a new-segment callback.",
        py::arg("params"), py::arg("callback") = py::none());

    m.def("parakeet_clear_new_segment_callback", &parakeet_clear_new_segment_callback,
        "Clear any previously assigned new-segment callback.",
        py::arg("params"));

    m.def("parakeet_assign_encoder_begin_callback",
            [](parakeet_full_params * params, py::object callback) {
                parakeet_assign_encoder_begin_callback(params, callback);
            },
            "Assign an encoder-begin callback.",
            py::arg("params"), py::arg("callback") = py::none());

    m.def("parakeet_clear_encoder_begin_callback", &parakeet_clear_encoder_begin_callback,
            "Clear any previously assigned encoder-begin callback.",
            py::arg("params"));

    m.def("parakeet_assign_abort_callback",
            [](parakeet_full_params * params, py::object callback) {
                parakeet_assign_abort_callback(params, callback);
            },
            "Assign an abort callback that returns True to stop processing.",
            py::arg("params"), py::arg("callback") = py::none());

    m.def("parakeet_clear_abort_callback", &parakeet_clear_abort_callback, "Clear any previously assigned abort callback.",
            py::arg("params"));

    m.def("parakeet_log_set",
        [](py::object callback) {
            parakeet_log_set_wrapper(callback);
        },
        "Assign a Python log callback or None to restore the default logger.",
        py::arg("callback") = py::none());

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
