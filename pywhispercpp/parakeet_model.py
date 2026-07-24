#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Thin Python helper for whisper.cpp Parakeet (`parakeet.h`).

Models are local GGUF files (e.g. https://huggingface.co/ggml-org/parakeet-GGUF).

Example::

    from pywhispercpp import ParakeetModel

    model = ParakeetModel("/path/to/ggml-parakeet-tdt-0.6b-v3-q5_0.bin", n_threads=4)
    for seg in model.transcribe("audio.wav"):
        print(seg.text)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

import _pywhispercpp as pw
from pywhispercpp.model import Model, Segment


class ParakeetModel:
    """
    Load a local Parakeet GGUF and run ``parakeet_full``.

    Reuses :class:`~pywhispercpp.model.Segment` and ``Model._load_audio`` so we
    do not duplicate wav/ffmpeg decoding.
    """

    def __init__(
        self,
        model_path: str,
        n_threads: Optional[int] = None,
        use_gpu: bool = True,
        gpu_device: int = 0,
        **params,
    ):
        """
        :param model_path: Path to a Parakeet GGUF model file.
        :param n_threads: Inference threads (default min(4, CPU count)).
        :param use_gpu: Forwarded to ``parakeet_context_params.use_gpu``.
        :param gpu_device: Forwarded to ``parakeet_context_params.gpu_device``.
        :param params: Extra fields on ``parakeet_full_params`` (e.g. ``offset_ms``).
        """
        self._ctx = None
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Parakeet model not found: {model_path}. "
                "Download a GGUF from https://huggingface.co/ggml-org/parakeet-GGUF"
            )
        self.model_path = str(path.resolve())

        cparams = pw.parakeet_context_default_params()
        cparams.use_gpu = use_gpu
        cparams.gpu_device = gpu_device

        self._params = pw.parakeet_full_default_params(
            pw.parakeet_sampling_strategy.PARAKEET_SAMPLING_GREEDY
        )
        self._params.n_threads = int(
            n_threads if n_threads is not None else min(4, os.cpu_count() or 4)
        )
        for key, value in params.items():
            setattr(self._params, key, value)

        self._ctx = pw.parakeet_init_from_file_with_params(self.model_path, cparams)
        if self._ctx is None:
            raise RuntimeError(f"Failed to load Parakeet model: {self.model_path}")

    def transcribe(
        self,
        media: Union[str, np.ndarray],
        **params,
    ) -> List[Segment]:
        """
        Transcribe a media path or float32 mono PCM at 16 kHz.

        :param media: File path (wav preferred; other formats need ffmpeg) or
                      1-D float32 ``numpy.ndarray``.
        :param params: Overrides for ``parakeet_full_params`` for this call.
        :return: List of :class:`~pywhispercpp.model.Segment`.
        """
        if isinstance(media, np.ndarray):
            audio = np.ascontiguousarray(media, dtype=np.float32).ravel()
        else:
            if not Path(media).exists():
                raise FileNotFoundError(media)
            audio = Model._load_audio(str(media))

        for key, value in params.items():
            setattr(self._params, key, value)

        ret = pw.parakeet_full(self._ctx, self._params, audio, int(audio.size))
        if ret != 0:
            raise RuntimeError(f"parakeet_full failed with code {ret}")

        n = pw.parakeet_full_n_segments(self._ctx)
        segments: List[Segment] = []
        for i in range(n):
            t0 = pw.parakeet_full_get_segment_t0(self._ctx, i)
            t1 = pw.parakeet_full_get_segment_t1(self._ctx, i)
            text = pw.parakeet_full_get_segment_text(self._ctx, i).decode(
                "utf-8", errors="replace"
            ).strip()
            segments.append(Segment(t0, t1, text))
        return segments

    def __del__(self):
        ctx = getattr(self, "_ctx", None)
        if ctx is not None:
            try:
                pw.parakeet_free(ctx)
            except Exception:
                pass
            self._ctx = None
