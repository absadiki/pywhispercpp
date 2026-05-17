#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a simple Python API on-top of the C-style
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) API.
"""
import importlib.metadata
import logging
import shutil
import sys
from pathlib import Path
from time import time
from typing import Any, Union, Callable, List, TextIO, Tuple, Optional, Dict, TypedDict
import _pywhispercpp as pw
import numpy as np
import pywhispercpp.utils as utils
import pywhispercpp.constants as constants
import subprocess
import os
import tempfile
import wave

__author__ = "absadiki"
__copyright__ = "Copyright 2023, "
__license__ = "MIT"
__version__ = importlib.metadata.version('pywhispercpp')

logger = logging.getLogger(__name__)


class ContextParams(TypedDict, total=False):
    use_gpu: bool
    flash_attn: bool
    gpu_device: int
    dtw_token_timestamps: bool
    dtw_aheads_preset: int
    dtw_n_top: int
    dtw_mem_size: int


_CONTEXT_PARAM_KEYS = frozenset(ContextParams.__annotations__)


class Segment:
    """
    A small class representing a transcription segment
    """

    def __init__(self, t0: int, t1: int, text: str, probability: float = np.nan):
        """
        :param t0: start time
        :param t1: end time
        :param text: text
        :param probability: Confidence score for the segment, computed as the geometric mean of
            the token probabilities for the segment (NaN if not calculated).
            This makes it interpretable as a probability in [0, 1].
        """
        self.t0 = t0
        self.t1 = t1
        self.text = text
        self.probability = probability

    def __str__(self):
        return f"t0={self.t0}, t1={self.t1}, text={self.text}, probability={self.probability}"

    def __repr__(self):
        return str(self)


class Model:
    """
    This classes defines a Whisper.cpp model.

    Example usage.
    ```python
    model = Model('base.en', n_threads=6)
    segments = model.transcribe('file.mp3')
    for segment in segments:
        print(segment.text)
    ```
    """

    _new_segment_callback = None

    def __init__(self,
                 model: str = 'tiny',
                 models_dir: Optional[str] = None,
                 params_sampling_strategy: int = 0,
                 redirect_whispercpp_logs_to: Union[bool, TextIO, str, None] = False,
                 use_openvino: bool = False,
                 openvino_model_path: Optional[str] = None,
                 openvino_device: str = 'CPU',
                 openvino_cache_dir: Optional[str] = None,
                 context_params: Optional[ContextParams] = None,
                 **params):
        """
        :param model: model name, default `tiny`, or a direct path to a ggml model file.
        :param models_dir: directory containing model files; if omitted, uses `MODELS_DIR` unless `model`
                           is already a direct file path.
        :param params_sampling_strategy: sampling strategy selector; `0` uses greedy decoding and any
                                         other value uses beam search.
        :param redirect_whispercpp_logs_to: log redirection target. Use `False` for no redirection, `None`
                                            for `/dev/null`, a file path string, or `sys.stdout`/`sys.stderr`.
        :param use_openvino: whether to initialize the OpenVINO encoder backend.
        :param openvino_model_path: path to the OpenVINO model directory or files.
        :param openvino_device: OpenVINO device name, default `CPU`.
        :param openvino_cache_dir: OpenVINO cache directory.
        :param context_params: optional whisper context loader params. Accepted keys are `use_gpu`,
                               `flash_attn`, `gpu_device`, `dtw_token_timestamps`,
                               `dtw_aheads_preset`, `dtw_n_top`, and `dtw_mem_size`. Omitted keys inherit
                               from `whisper_context_default_params()`.
        :param params: decode parameters forwarded to `whisper_full_params`.
            Supported keys:
            - `n_threads`: number of inference threads. Default is `min(4, hardware_concurrency())`.
            - `n_max_text_ctx`: max prompt-text tokens carried into the decoder. Default `16384`.
            - `offset_ms`: audio start offset in milliseconds. Default `0`.
            - `duration_ms`: audio duration to process in milliseconds. Default `0`.
            - `translate`: translate output to English. Default `False`.
            - `no_context`: disable reuse of past transcription context. Default `True`.
            - `no_timestamps`: disable timestamp generation. Default `False`.
            - `single_segment`: force a single output segment. Default `False`.
            - `print_special`: print special tokens. Default `False`.
            - `print_progress`: print progress information. Default `True`.
            - `print_realtime`: print realtime output from whisper.cpp. Default `False`.
            - `print_timestamps`: print timestamps during realtime output. Default `True`.
            - `token_timestamps`: enable token-level timestamps. Default `False`.
            - `thold_pt`: token timestamp probability threshold. Default `0.01`.
            - `thold_ptsum`: token timestamp sum threshold. Default `0.01`.
            - `max_len`: max segment length in characters. Default `0`.
            - `split_on_word`: split on words when `max_len` is used. Default `False`.
            - `max_tokens`: max tokens per segment. Default `0`.
            - `debug_mode`: enable whisper.cpp debug mode. Default `False`.
            - `audio_ctx`: override audio context size. Default `0`.
            - `tdrz_enable`: enable tinydiarize speaker-turn detection. Default `False`.
            - `initial_prompt`: initial text prompt prepended before decoding. Default `None`.
            - `grammar`: GBNF grammar text or path to a grammar file. Default `None`.
            - `grammar_rule`: top-level grammar rule name. Default `root` when grammar is used.
            - `prompt_tokens`: explicit prompt token sequence. Default `None`.
            - `prompt_n_tokens`: number of prompt tokens. Default `0`.
            - `carry_initial_prompt`: prepend the initial prompt to each decode window. Default `False`.
            - `language`: language code. Default `en`.
            - `detect_language`: enable automatic language detection during transcription. Default `False`.
            - `suppress_blank`: suppress blank outputs. Default `True`.
            - `suppress_non_speech_tokens`: Python alias for `suppress_nst`. Default `False`.
            - `suppress_nst`: suppress non-speech tokens. Default `False`.
            - `temperature`: initial decoding temperature. Default `0.0`.
            - `max_initial_ts`: maximum initial timestamp. Default `1.0`.
            - `length_penalty`: length penalty. Default `-1.0`.
            - `temperature_inc`: fallback temperature increment. Default `0.2`.
            - `entropy_thold`: entropy threshold. Default `2.4`.
            - `logprob_thold`: logprob threshold. Default `-1.0`.
            - `no_speech_thold`: no-speech threshold. Default `0.6`.
            - `grammar_penalty`: penalty applied to non-grammar tokens. Default `100.0`.
            - `greedy`: greedy-decoder settings, typically `{"best_of": 5}`.
            - `beam_search`: beam-search settings, schema default `{"beam_size": 5, "patience": -1.0}`.
            - `vad`: enable VAD. Default `False`.
            - `vad_model_path`: path to the VAD model. Default `None`.
        """
        self.model_path = utils.resolve_model_path(model, models_dir)
        self._ctx = None
        self._context_params = self._resolve_context_params(context_params)
        self._sampling_strategy = pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY if params_sampling_strategy == 0 else \
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH
        self._params = pw.whisper_full_default_params(self._sampling_strategy)
        # assign params
        self.params = params
        self._set_params(params)
        self.redirect_whispercpp_logs_to = redirect_whispercpp_logs_to
        self.use_openvino = use_openvino
        self.openvino_model_path = openvino_model_path
        self.openvino_device = openvino_device
        self.openvino_cache_dir = openvino_cache_dir
        # init the model
        self._init_model()

    def transcribe(self,
                   media: Union[str, np.ndarray],
                   n_processors: Optional[int] = None,
                   new_segment_callback: Optional[Callable[[Segment], None]] = None,
                   abort_callback: Optional[Callable[[], bool]] = None,
                   **params) -> List[Segment]:
        """
        Transcribes the media provided as input and returns list of `Segment` objects.
        Accepts a media_file path (audio/video) or a raw numpy array.

        :param media: Media file path or a numpy array
        :param n_processors: number of worker processes for `whisper_full_parallel`. If omitted, runs a
                     single-process `whisper_full()` decode.
        :param new_segment_callback: callback invoked for each newly produced `Segment` during decoding.
        :param abort_callback: callback function returning True to abort an in-flight transcription early
        :param params: keyword arguments for different whisper.cpp parameters; these override the model's
                       active decode params for this call
        :param extract_probability: If True, calculates the geometric mean of token probabilities for each segment,
            providing a confidence score interpretable as a probability in [0, 1].
        :return: List of transcription segments
        """
        if isinstance(media, np.ndarray):
            audio = media
        else:
            if not Path(media).exists():
                raise FileNotFoundError(media)
            audio = self._load_audio(media)

        # Handle extract_probability parameter
        self.extract_probability = params.pop('extract_probability', False)

        # update params if any
        self._set_params(params)

        # setting up callback
        if new_segment_callback:
            Model._new_segment_callback = new_segment_callback
            pw.assign_new_segment_callback(self._params, Model.__call_new_segment_callback)

        pw.assign_abort_callback(self._params, abort_callback)

        # run inference
        start_time = time()
        logger.info("Transcribing ...")
        res = self._transcribe(audio, n_processors=n_processors)
        end_time = time()
        logger.info(f"Inference time: {end_time - start_time:.3f} s")
        return res

    @staticmethod
    def _get_segments(ctx, start: int, end: int, extract_probability: bool = False) -> List[Segment]:
        """
        Helper function to get generated segments between `start` and `end`

        :param ctx: whisper context
        :param start: start index
        :param end: end index
        :param extract_probability: whether to calculate token probabilities

        :return: list of segments
        """
        n = pw.whisper_full_n_segments(ctx)
        assert end <= n, f"{end} > {n}: `End` index must be less or equal than the total number of segments"
        res = []
        for i in range(start, end):
            t0 = pw.whisper_full_get_segment_t0(ctx, i)
            t1 = pw.whisper_full_get_segment_t1(ctx, i)
            bytes = pw.whisper_full_get_segment_text(ctx, i)
            text = bytes.decode('utf-8', errors='replace')

            avg_prob = np.nan

            # Only calculate probabilities if requested
            if extract_probability:
                n_tokens = pw.whisper_full_n_tokens(ctx, i)
                if n_tokens == 1:
                    avg_prob = pw.whisper_full_get_token_p(ctx, i, 0)
                elif n_tokens > 1:
                    total_logprob = 0.0
                    for j in range(n_tokens):
                        total_logprob += np.log(pw.whisper_full_get_token_p(ctx, i, j))
                    avg_prob = np.exp(total_logprob / n_tokens)
                else:
                    avg_prob = np.nan

            res.append(Segment(t0, t1, text.strip(), probability=float(avg_prob)))
        return res

    def get_params(self) -> dict:
        """
        Returns a `dict` representation of the actual params

        :return: params dict
        """
        res = {}
        for param in dir(self._params):
            if param.startswith('__'):
                continue
            try:
                res[param] = getattr(self._params, param)
            except Exception:
                # ignore callback functions
                continue
        return res

    @staticmethod
    def get_params_schema() -> dict:
        """
        A simple link to ::: constants.PARAMS_SCHEMA
        :return: dict of params schema
        """
        return constants.PARAMS_SCHEMA

    @staticmethod
    def lang_max_id() -> int:
        """
        Largest language id (i.e. number of available languages - 1)
        Direct binding to whisper.cpp/lang_max_id
        :return:
        """
        return pw.whisper_lang_max_id()

    def print_timings(self) -> None:
        """
        Direct binding to whisper.cpp/whisper_print_timings

        :return: None
        """
        pw.whisper_print_timings(self._ctx)

    @staticmethod
    def system_info() -> None:
        """
        Direct binding to whisper.cpp/whisper_print_system_info

        :return: None
        """
        return pw.whisper_print_system_info()

    @staticmethod
    def available_languages() -> List[str]:
        """
        Returns a list of supported language codes

        :return: list of supported language codes
        """
        n = pw.whisper_lang_max_id()
        res = []
        for i in range(n+1):
            res.append(pw.whisper_lang_str(i))
        return res

    @staticmethod
    def _resolve_context_params(context_params: Optional[ContextParams]):
        if context_params is None:
            return None

        if not isinstance(context_params, dict):
            raise TypeError("context_params must be a ContextParams dict or None")

        unknown_keys = sorted(set(context_params) - _CONTEXT_PARAM_KEYS)
        if unknown_keys:
            raise TypeError(
                f"Unknown context_params keys: {', '.join(unknown_keys)}"
            )

        resolved = pw.whisper_context_default_params()
        for key, value in context_params.items():
            setattr(resolved, key, value)
        return resolved

    @staticmethod
    def _normalize_params(kwargs: dict) -> dict:
        normalized = dict(kwargs)

        if 'suppress_non_speech_tokens' in normalized and 'suppress_nst' not in normalized:
            normalized['suppress_nst'] = normalized.pop('suppress_non_speech_tokens')

        return normalized

    def _init_model(self) -> None:
        """
        Private method to initialize the method from the bindings, it will be called automatically from the __init__
        :return:
        """
        logger.info("Initializing the model ...")
        with utils.redirect_stderr(to=self.redirect_whispercpp_logs_to):
            if self._context_params is None:
                self._ctx = pw.whisper_init_from_file(self.model_path)
            else:
                self._ctx = pw.whisper_init_from_file_with_params(self.model_path, self._context_params)
            if self.use_openvino:
                pw.whisper_ctx_init_openvino_encoder(self._ctx, self.openvino_model_path, self.openvino_device, self.openvino_cache_dir)



    def _set_params(self, kwargs: dict) -> None:
        """
        Private method to set the kwargs params to the `Params` class
        :param kwargs: dict like object for the different params
        :return: None
        """
        normalized = self._normalize_params(kwargs)
        prompt_tokens = normalized.pop('prompt_tokens', None) if 'prompt_tokens' in normalized else None
        grammar = normalized.pop('grammar', None) if 'grammar' in normalized else None
        grammar_rule = normalized.pop('grammar_rule', None) if 'grammar_rule' in normalized else None
        grammar_penalty = normalized.get('grammar_penalty', self._params.grammar_penalty)

        for param, value in normalized.items():
            setattr(self._params, param, value)

        if 'prompt_tokens' in kwargs:
            self._params.set_prompt_tokens(prompt_tokens)

        if 'grammar' in kwargs:
            self._params.set_grammar(grammar, grammar_rule, grammar_penalty)

    def _transcribe(self, audio: np.ndarray, n_processors: Optional[int] = None):
        """
        Private method to call the whisper.cpp/whisper_full function

        :param audio: numpy array of audio data
        :param n_processors: if not None, it will run whisper.cpp/whisper_full_parallel with n_processors
        :return:
        """

        if n_processors:
            pw.whisper_full_parallel(self._ctx, self._params, audio, audio.size, n_processors)
        else:
            pw.whisper_full(self._ctx, self._params, audio, audio.size)
        n = pw.whisper_full_n_segments(self._ctx)
        res = Model._get_segments(self._ctx, 0, n, self.extract_probability)
        return res

    @staticmethod
    def __call_new_segment_callback(ctx, n_new, user_data) -> None:
        """
        Internal new_segment_callback, it just calls the user's callback with the `Segment` object
        :param ctx: whisper.cpp ctx param
        :param n_new: whisper.cpp n_new param
        :param user_data: whisper.cpp user_data param
        :return: None
        """
        n = pw.whisper_full_n_segments(ctx)
        start = n - n_new
        res = Model._get_segments(ctx, start, n, False)
        for segment in res:
            if Model._new_segment_callback is not None:
                Model._new_segment_callback(segment)

    @staticmethod
    def _load_audio(media_file_path: str) -> np.ndarray:
        """
         Helper method to return a `np.array` object from a media file
         If the media file is not a WAV file, it will try to convert it using ffmpeg

        :param media_file_path: Path of the media file
        :return: Numpy array
        """

        def wav_to_np(file_path):
            with wave.open(file_path, 'rb') as wf:
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                num_frames = wf.getnframes()

                if num_channels not in (1, 2):
                    raise Exception(f"WAV file must be mono or stereo")

                if sample_rate != pw.WHISPER_SAMPLE_RATE:
                    raise Exception(f"WAV file must be {pw.WHISPER_SAMPLE_RATE} Hz")

                if sample_width != 2:
                    raise Exception(f"WAV file must be 16-bit")

                raw = wf.readframes(num_frames)
                wf.close()
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                n = num_frames
                if num_channels == 1:
                    pcmf32 = audio / 32768.0
                else:
                    audio = audio.reshape(-1, 2)
                    # Averaging the two channels
                    pcmf32 = (audio[:, 0] + audio[:, 1]) / 65536.0
                return pcmf32

        if media_file_path.endswith('.wav'):
            return wav_to_np(media_file_path)
        else:
            if shutil.which('ffmpeg') is None:
                raise Exception(
                    "FFMPEG is not installed or not in PATH. Please install it, or provide a WAV file or a NumPy array instead!")

            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file_path = temp_file.name
            temp_file.close()
            try:
                subprocess.run([
                    'ffmpeg', '-i', media_file_path, '-ac', '1', '-ar', '16000',
                    temp_file_path, '-y'
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return wav_to_np(temp_file_path)
            finally:
                os.remove(temp_file_path)

    def auto_detect_language(self, media: Union[str, np.ndarray], offset_ms: Optional[int] = None, n_threads: Optional[int] = None) -> Tuple[Tuple[str, np.float32], Dict[str, np.float32]]:
        """
        Automatic language detection using whisper.cpp/whisper_pcm_to_mel and whisper.cpp/whisper_lang_auto_detect

        :param media: Media file path or a numpy array
        :param offset_ms: offset in milliseconds; when omitted, uses the model's current `offset_ms`
        :param n_threads: number of threads to use; when omitted, uses the model's current `n_threads`
        :return: ((detected_language, probability), probabilities for all languages)
        """
        if isinstance(media, np.ndarray):
            audio = media
        else:
            if not Path(media).exists():
                raise FileNotFoundError(media)
            audio = self._load_audio(media)

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

    def __del__(self):
        """
        Free up resources
        :return: None
        """
        if self._ctx is not None:
            pw.whisper_free(self._ctx)