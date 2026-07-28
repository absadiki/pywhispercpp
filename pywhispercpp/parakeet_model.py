#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python API for Parakeet TDT models
"""
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from time import time
from typing import List, Optional, Union

import _pywhispercpp as pw
import numpy as np

import pywhispercpp.utils as utils
from pywhispercpp.base import BaseModel, Segment

logger = logging.getLogger(__name__)


@dataclass
class ParakeetContextParams:
    """
    Parameters for model context initialization.
    """
    use_gpu: Optional[bool] = None
    """Use GPU for inference."""

    gpu_device: Optional[int] = None
    """GPU device index."""

    def apply(self, c_params) -> None:
        """Apply non-None fields to the C ``parakeet_context_params`` struct."""
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                setattr(c_params, f.name, val)


@dataclass
class ParakeetParams:
    """
    Parameters for Parakeet transcription.

    These map directly to ``parakeet_full_params`` fields in the C library.
    """
    n_threads: Optional[int] = None
    """Number of inference threads."""

    offset_ms: Optional[int] = None
    """Start offset in milliseconds."""

    duration_ms: Optional[int] = None
    """Audio duration to process in milliseconds."""

    no_context: Optional[bool] = None
    """Disable reuse of past transcription context."""

    audio_ctx: Optional[int] = None
    """Override audio context size (0 = use default)."""

    def apply(self, c_params) -> None:
        """Apply non-None fields to the C ``parakeet_full_params`` struct."""
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                setattr(c_params, f.name, val)


class ParakeetModel(BaseModel):
    """
    A Parakeet TDT model backed by parakeet.cpp.

    Example usage::

        model = ParakeetModel('path/to/parakeet.gguf')
        segments = model.transcribe('audio.wav')
        for segment in segments:
            print(segment.text)
    """

    def __init__(self,
                 model: str,
                 models_dir: Optional[str] = None,
                 redirect_logs_to: Union[bool, None] = False,
                 context_params: Optional[ParakeetContextParams] = None,
                 params: Optional[ParakeetParams] = None,
                 **kwargs):
        """
        :param model: path to a Parakeet GGUF model file, or a model name to resolve/download.
        :param models_dir: directory to search for local model files. Defaults to ``MODELS_DIR``
            from constants. If the model is not found locally, it will be downloaded and stored
            here. Ignored when *model* is a direct file path.
        :param redirect_logs_to: log redirection target. ``False`` for no redirection, ``None`` for /dev/null.
        :param context_params: optional context params (GPU settings).
        :param params: a ``ParakeetParams`` instance for default decode settings.
        :param kwargs: override params as keyword arguments (e.g. ``n_threads=8``).
        """
        self.model_path = utils.resolve_model_path(model, models_dir)
        self._ctx = None
        self._context_params = self._resolve_context_params(context_params)
        self._params = pw.parakeet_full_default_params(
            pw.parakeet_sampling_strategy.PARAKEET_SAMPLING_GREEDY
        )

        if params is None:
            params = ParakeetParams(**kwargs)
        params.apply(self._params)

        self._redirect_logs_to = redirect_logs_to
        self._new_segment_callback = None
        self._init_model()

    def transcribe(self,
                   media: Union[str, np.ndarray],
                   new_segment_callback=None,
                   abort_callback=None,
                   params: Optional[ParakeetParams] = None,
                   **kwargs) -> List[Segment]:
        """
        Transcribe audio and return a list of ``Segment`` objects.

        :param media: file path or numpy array of audio data.
        :param new_segment_callback: callback invoked for each new segment.
        :param abort_callback: callback returning ``True`` to abort.
        :param params: optional ``ParakeetParams`` overrides for this call only.
        :param kwargs: override params as keyword arguments.
        :return: list of transcription segments.
        """
        if isinstance(media, np.ndarray):
            audio = media
        else:
            if not Path(media).exists():
                raise FileNotFoundError(media)
            audio = utils.load_audio(media, sample_rate=pw.PARAKEET_SAMPLE_RATE)

        if params is not None:
            params.apply(self._params)
        elif kwargs:
            ParakeetParams(**kwargs).apply(self._params)

        self._new_segment_callback = new_segment_callback
        pw.parakeet_assign_new_segment_callback(
            self._params,
            self._on_new_segment if new_segment_callback is not None else None,
        )

        pw.parakeet_assign_abort_callback(self._params, abort_callback)

        start_time = time()
        logger.info("Transcribing ...")
        pw.parakeet_full(self._ctx, self._params, audio, audio.size)
        n = pw.parakeet_full_n_segments(self._ctx)
        segments = self._get_segments(self._ctx, 0, n)
        end_time = time()
        logger.info(f"Inference time: {end_time - start_time:.3f} s")
        return segments

    def print_timings(self) -> None:
        pw.parakeet_print_timings(self._ctx)

    def print_system_info(self) -> None:
        pw.parakeet_print_system_info()

    def _init_model(self) -> None:
        logger.info("Initializing parakeet model ...")
        with utils.redirect_stderr(to=self._redirect_logs_to):
            self._ctx = pw.parakeet_init_from_file_with_params(
                self.model_path, self._context_params
            )

    def _on_new_segment(self, ctx, n_new, user_data=None) -> None:
        n = pw.parakeet_full_n_segments(ctx)
        start = n - n_new
        segments = self._get_segments(ctx, start, n)
        for segment in segments:
            if self._new_segment_callback is not None:
                self._new_segment_callback(segment)

    @staticmethod
    def _get_segments(ctx, start: int, end: int) -> List[Segment]:
        n = pw.parakeet_full_n_segments(ctx)
        assert end <= n, f"{end} > {n}: end index exceeds segment count"
        res = []
        for i in range(start, end):
            t0 = pw.parakeet_full_get_segment_t0(ctx, i)
            t1 = pw.parakeet_full_get_segment_t1(ctx, i)
            text = pw.parakeet_full_get_segment_text(ctx, i)
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            res.append(Segment(t0, t1, text.strip()))
        return res

    @staticmethod
    def _resolve_context_params(context_params: Optional[ParakeetContextParams]):
        resolved = pw.parakeet_context_default_params()
        if context_params is not None:
            context_params.apply(resolved)
        return resolved

    def __del__(self):
        if self._ctx is not None:
            pw.parakeet_free(self._ctx)
