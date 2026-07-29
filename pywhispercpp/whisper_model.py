#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python API for Whisper.cpp models
"""
import logging
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from time import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import _pywhispercpp as pw
import numpy as np

import pywhispercpp.utils as utils
from pywhispercpp.base import BaseModel, Segment

logger = logging.getLogger(__name__)


@dataclass
class WhisperContextParams:
    """
    Parameters for whisper context initialization.
    """
    use_gpu: Optional[bool] = None
    """Use GPU for inference."""

    flash_attn: Optional[bool] = None
    """Enable Flash Attention."""

    gpu_device: Optional[int] = None
    """GPU device index."""

    dtw_token_timestamps: Optional[bool] = None
    """Enable DTW token timestamps."""

    dtw_aheads_preset: Optional[int] = None
    """DTW aheads preset."""

    dtw_n_top: Optional[int] = None
    """DTW n_top."""

    dtw_mem_size: Optional[int] = None
    """DTW memory size."""

    def apply(self, c_params) -> None:
        """Apply non-None fields to the C ``whisper_context_params`` struct."""
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                setattr(c_params, f.name, val)


@dataclass
class WhisperParams:
    """
    Parameters for Whisper transcription.

    These map directly to ``whisper_full_params`` fields in the C-API.
    """
    n_threads: Optional[int] = None
    """Number of inference threads."""

    n_max_text_ctx: Optional[int] = None
    """Max prompt-text tokens carried into the decoder."""

    offset_ms: Optional[int] = None
    """Start offset in milliseconds."""

    duration_ms: Optional[int] = None
    """Audio duration to process in milliseconds."""

    translate: Optional[bool] = None
    """Translate output to English."""

    no_context: Optional[bool] = None
    """Disable reuse of past transcription context."""

    no_timestamps: Optional[bool] = None
    """Disable timestamp generation."""

    single_segment: Optional[bool] = None
    """Force single segment output."""

    print_special: Optional[bool] = None
    """Print special tokens."""

    print_progress: Optional[bool] = None
    """Print progress information."""

    print_realtime: Optional[bool] = None
    """Print realtime output."""

    print_timestamps: Optional[bool] = None
    """Print timestamps during realtime output."""

    token_timestamps: Optional[bool] = None
    """Enable token-level timestamps."""

    thold_pt: Optional[float] = None
    """Token timestamp probability threshold."""

    thold_ptsum: Optional[float] = None
    """Token timestamp sum threshold."""

    max_len: Optional[int] = None
    """Max segment length in characters."""

    split_on_word: Optional[bool] = None
    """Split on words when max_len is used."""

    max_tokens: Optional[int] = None
    """Max tokens per segment (0 = no limit)."""

    debug_mode: Optional[bool] = None
    """Enable whisper.cpp debug mode."""

    audio_ctx: Optional[int] = None
    """Override audio context size (0 = use default)."""

    tdrz_enable: Optional[bool] = None
    """Enable tinydiarize speaker-turn detection."""

    initial_prompt: Optional[str] = None
    """Initial text prompt prepended before decoding."""

    prompt_tokens: Optional[Tuple] = None
    """Explicit prompt token sequence."""

    prompt_n_tokens: Optional[int] = None
    """Number of prompt tokens."""

    carry_initial_prompt: Optional[bool] = None
    """Prepend the initial prompt to each decode window."""

    language: Optional[str] = None
    """Language code (e.g. ``"en"``, ``"de"``)."""

    detect_language: Optional[bool] = None
    """Enable automatic language detection."""

    suppress_blank: Optional[bool] = None
    """Suppress blank outputs."""

    suppress_non_speech_tokens: Optional[bool] = None
    """Alias for suppress_nst."""

    suppress_nst: Optional[bool] = None
    """Suppress non-speech tokens."""

    suppress_regex: Optional[str] = None
    """Regex pattern used to suppress matching text."""

    temperature: Optional[float] = None
    """Initial decoding temperature."""

    max_initial_ts: Optional[float] = None
    """Maximum initial timestamp."""

    length_penalty: Optional[float] = None
    """Length penalty."""

    temperature_inc: Optional[float] = None
    """Fallback temperature increment."""

    entropy_thold: Optional[float] = None
    """Entropy threshold."""

    logprob_thold: Optional[float] = None
    """Logprob threshold."""

    no_speech_thold: Optional[float] = None
    """No-speech threshold."""

    greedy: Optional[dict] = None
    """Greedy decoder settings (e.g. ``{"best_of": 5}``)."""

    beam_search: Optional[dict] = None
    """Beam search settings (e.g. ``{"beam_size": -1, "patience": -1.0}``)."""

    vad: Optional[bool] = None
    """Enable VAD."""

    vad_model_path: Optional[str] = None
    """Path to the VAD model."""

    def apply(self, c_params) -> None:
        """Apply non-None fields to the C ``whisper_full_params`` struct."""
        if self.suppress_non_speech_tokens is not None:
            c_params.suppress_nst = self.suppress_non_speech_tokens

        if self.prompt_tokens is not None:
            c_params.set_prompt_tokens(self.prompt_tokens)

        for f in fields(self):
            if f.name in ('suppress_non_speech_tokens', 'prompt_tokens'):
                continue
            val = getattr(self, f.name)
            if val is not None:
                setattr(c_params, f.name, val)


class WhisperModel(BaseModel):
    """
    A Whisper model Python API on top of whisper.cpp C-API.

    Example usage::

        model = WhisperModel('path/to/tiny.bin', params=WhisperParams(n_threads=8))
        segments = model.transcribe('audio.wav')
        for segment in segments:
            print(segment.text)

    Or use a model name to download automatically::

        model = WhisperModel(WhisperModel.AvailableModels.TINY)
    """

    class AvailableModels(StrEnum):
        """Available Whisper model names for auto-download."""
        TINY = "tiny"
        TINY_Q5_1 = "tiny-q5_1"
        TINY_Q8_0 = "tiny-q8_0"
        TINY_EN = "tiny.en"
        TINY_EN_Q5_1 = "tiny.en-q5_1"
        TINY_EN_Q8_0 = "tiny.en-q8_0"
        BASE = "base"
        BASE_Q5_1 = "base-q5_1"
        BASE_Q8_0 = "base-q8_0"
        BASE_EN = "base.en"
        BASE_EN_Q5_1 = "base.en-q5_1"
        BASE_EN_Q8_0 = "base.en-q8_0"
        SMALL = "small"
        SMALL_Q5_1 = "small-q5_1"
        SMALL_Q8_0 = "small-q8_0"
        SMALL_EN = "small.en"
        SMALL_EN_Q5_1 = "small.en-q5_1"
        SMALL_EN_Q8_0 = "small.en-q8_0"
        MEDIUM = "medium"
        MEDIUM_Q5_0 = "medium-q5_0"
        MEDIUM_Q8_0 = "medium-q8_0"
        MEDIUM_EN = "medium.en"
        MEDIUM_EN_Q5_0 = "medium.en-q5_0"
        MEDIUM_EN_Q8_0 = "medium.en-q8_0"
        LARGE_V1 = "large-v1"
        LARGE_V2 = "large-v2"
        LARGE_V2_Q5_0 = "large-v2-q5_0"
        LARGE_V2_Q8_0 = "large-v2-q8_0"
        LARGE_V3 = "large-v3"
        LARGE_V3_Q5_0 = "large-v3-q5_0"
        LARGE_V3_TURBO = "large-v3-turbo"
        LARGE_V3_TURBO_Q5_0 = "large-v3-turbo-q5_0"
        LARGE_V3_TURBO_Q8_0 = "large-v3-turbo-q8_0"

    _MODELS_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp"
    _MODELS_PREFIX_URL = "resolve/main/ggml"

    def __init__(self,
                 model: str = 'tiny',
                 models_dir: Optional[str] = None,
                 redirect_logs_to: Union[bool, None] = False,
                 context_params: Optional[WhisperContextParams] = None,
                 params: Optional[WhisperParams] = None,
                 sampling_strategy: int = 0,
                 use_openvino: bool = False,
                 openvino_model_path: Optional[str] = None,
                 openvino_device: str = 'CPU',
                 openvino_cache_dir: Optional[str] = None,
                 **kwargs):
        """
        :param model: model name (e.g. ``"tiny"``, ``"base.en"``) or a direct path to a ggml/gguf model file.
            Use ``WhisperModel.AvailableModels`` for available model names. If the model is not
            found locally, it will be downloaded automatically.
        :param models_dir: directory to search for local model files. Defaults to ``MODELS_DIR``.
            Ignored when *model* is a direct file path.
        :param redirect_logs_to: log redirection target. ``False`` for no redirection, ``None`` for /dev/null.
        :param context_params: optional ``WhisperContextParams`` for GPU settings.
        :param params: a ``WhisperParams`` instance for default decode settings.
        :param sampling_strategy: ``0`` for greedy, ``1`` for beam search.
        :param use_openvino: whether to initialize the OpenVINO encoder backend.
        :param openvino_model_path: path to the OpenVINO model directory or files.
        :param openvino_device: OpenVINO device name, default ``"CPU"``.
        :param openvino_cache_dir: OpenVINO cache directory.
        :param kwargs: override params as keyword arguments (e.g. ``n_threads=8``).
        """
        self.model_path = utils.resolve_model_path(
            model, models_dir,
            base_url=self._MODELS_BASE_URL,
            prefix_url=self._MODELS_PREFIX_URL,
        )
        self._ctx = None
        self._context_params = self._resolve_context_params(context_params)
        self._sampling_strategy = (
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
            if sampling_strategy == 0
            else pw.whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH
        )
        self._params = pw.whisper_full_default_params(self._sampling_strategy)

        if params is None:
            params = WhisperParams(**kwargs)
        params.apply(self._params)

        self._redirect_logs_to = redirect_logs_to
        self._use_openvino = use_openvino
        self._openvino_model_path = openvino_model_path
        self._openvino_device = openvino_device
        self._openvino_cache_dir = openvino_cache_dir
        self._new_segment_callback = None
        self._init_model()

    def transcribe(self,
                   media: Union[str, np.ndarray],
                   new_segment_callback: Optional[Callable[[Segment], None]] = None,
                   abort_callback: Optional[Callable[[], bool]] = None,
                   params: Optional[WhisperParams] = None,
                   n_processors: Optional[int] = None,
                   extract_probability: bool = False,
                   **kwargs) -> List[Segment]:
        """
        Transcribe audio and return a list of ``Segment`` objects.

        :param media: file path or numpy array of audio data.
        :param new_segment_callback: callback invoked for each new segment.
        :param abort_callback: callback returning ``True`` to abort.
        :param params: optional ``WhisperParams`` overrides for this call only.
        :param n_processors: number of worker processes for ``whisper_full_parallel``.
            If omitted, runs single-process ``whisper_full()``.
        :param extract_probability: if ``True``, computes geometric mean of token probabilities
            per segment as a confidence score in [0, 1].
        :param kwargs: override params as keyword arguments.
        :return: list of transcription segments.
        """
        if isinstance(media, np.ndarray):
            audio = media
        else:
            if not Path(media).exists():
                raise FileNotFoundError(media)
            audio = utils.load_audio(media, sample_rate=pw.WHISPER_SAMPLE_RATE)

        if params is not None:
            params.apply(self._params)
        elif kwargs:
            WhisperParams(**kwargs).apply(self._params)

        self._new_segment_callback = new_segment_callback
        pw.assign_new_segment_callback(
            self._params,
            self._on_new_segment if new_segment_callback is not None else None,
        )

        pw.assign_abort_callback(self._params, abort_callback)

        start_time = time()
        logger.info("Transcribing ...")
        if n_processors:
            pw.whisper_full_parallel(self._ctx, self._params, audio, audio.size, n_processors)
        else:
            pw.whisper_full(self._ctx, self._params, audio, audio.size)
        n = pw.whisper_full_n_segments(self._ctx)
        segments = self._get_segments(self._ctx, 0, n, extract_probability)
        end_time = time()
        logger.info(f"Inference time: {end_time - start_time:.3f} s")
        return segments

    def auto_detect_language(self,
                             media: Union[str, np.ndarray],
                             offset_ms: Optional[int] = None,
                             n_threads: Optional[int] = None) -> Tuple[Tuple[str, np.float32], Dict[str, np.float32]]:
        """
        Automatic language detection using ``whisper_pcm_to_mel`` and ``whisper_lang_auto_detect``.

        :param media: file path or numpy array of audio data.
        :param offset_ms: offset in milliseconds; when omitted, uses the model's current ``offset_ms``.
        :param n_threads: number of threads; when omitted, uses the model's current ``n_threads``.
        :return: ``((detected_language, probability), probabilities_for_all_languages)``.
        """
        if isinstance(media, np.ndarray):
            audio = media
        else:
            if not Path(media).exists():
                raise FileNotFoundError(media)
            audio = utils.load_audio(media, sample_rate=pw.WHISPER_SAMPLE_RATE)

        if offset_ms is None:
            offset_ms = self._params.offset_ms

        if n_threads is None:
            n_threads = self._params.n_threads

        pw.whisper_pcm_to_mel(self._ctx, audio, len(audio), n_threads)
        lang_count = self.lang_max_id() + 1
        probs = np.zeros(lang_count, dtype=np.float32)
        auto_detect = pw.whisper_lang_auto_detect(self._ctx, offset_ms, n_threads, probs)
        langs = self.available_languages()
        lang_probs = {langs[i]: probs[i] for i in range(lang_count)}
        return (langs[auto_detect], np.float32(probs[auto_detect])), lang_probs

    def print_timings(self) -> None:
        pw.whisper_print_timings(self._ctx)

    @staticmethod
    def print_system_info() -> None:
        pw.whisper_print_system_info()

    @staticmethod
    def lang_max_id() -> int:
        return pw.whisper_lang_max_id()

    @staticmethod
    def available_languages() -> List[str]:
        n = pw.whisper_lang_max_id()
        return [pw.whisper_lang_str(i) for i in range(n + 1)]

    def _init_model(self) -> None:
        logger.info("Initializing whisper model ...")
        with utils.redirect_stderr(to=self._redirect_logs_to):
            self._ctx = pw.whisper_init_from_file_with_params(
                self.model_path, self._context_params
            )
            if self._use_openvino:
                pw.whisper_ctx_init_openvino_encoder(
                    self._ctx, self._openvino_model_path,
                    self._openvino_device, self._openvino_cache_dir,
                )

    def _on_new_segment(self, ctx, n_new, user_data=None) -> None:
        n = pw.whisper_full_n_segments(ctx)
        start = n - n_new
        segments = self._get_segments(ctx, start, n, False)
        for segment in segments:
            if self._new_segment_callback is not None:
                self._new_segment_callback(segment)

    @staticmethod
    def _get_segments(ctx, start: int, end: int, extract_probability: bool = False) -> List[Segment]:
        n = pw.whisper_full_n_segments(ctx)
        assert end <= n, f"{end} > {n}: end index exceeds segment count"
        res = []
        for i in range(start, end):
            t0 = pw.whisper_full_get_segment_t0(ctx, i)
            t1 = pw.whisper_full_get_segment_t1(ctx, i)
            text = pw.whisper_full_get_segment_text(ctx, i)
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')

            avg_prob = np.nan
            if extract_probability:
                n_tokens = pw.whisper_full_n_tokens(ctx, i)
                if n_tokens == 1:
                    avg_prob = pw.whisper_full_get_token_p(ctx, i, 0)
                elif n_tokens > 1:
                    total_logprob = 0.0
                    for j in range(n_tokens):
                        total_logprob += np.log(pw.whisper_full_get_token_p(ctx, i, j))
                    avg_prob = np.exp(total_logprob / n_tokens)

            res.append(Segment(t0, t1, text.strip(), probability=float(avg_prob)))
        return res

    @staticmethod
    def _resolve_context_params(context_params: Optional[WhisperContextParams]):
        resolved = pw.whisper_context_default_params()
        if context_params is not None:
            context_params.apply(resolved)
        return resolved

    def __del__(self):
        if self._ctx is not None:
            pw.whisper_free(self._ctx)
