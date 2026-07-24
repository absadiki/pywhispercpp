#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
High-level Python API for whisper.cpp Parakeet models.

Parakeet is a separate C surface from Whisper (`parakeet.h`). Models are GGUF
checkpoints, for example from https://huggingface.co/ggml-org/parakeet-GGUF .

Example::

    from pywhispercpp import ParakeetModel

    model = ParakeetModel("/path/to/ggml-parakeet-tdt-0.6b-v3-q5_0.bin", n_threads=4)
    segments = model.transcribe("audio.wav")
    for seg in segments:
        print(seg.t0, seg.t1, seg.text)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List, Optional, TypedDict, Union

import numpy as np

import _pywhispercpp as pw
from pywhispercpp.model import Segment

logger = logging.getLogger(__name__)


class ParakeetContextParams(TypedDict, total=False):
    use_gpu: bool
    gpu_device: int


_CONTEXT_PARAM_KEYS = frozenset(ParakeetContextParams.__annotations__)


class ParakeetModel:
    """
    Thin wrapper around the Parakeet C API exposed by ``_pywhispercpp``.

    Unlike :class:`pywhispercpp.model.Model`, this does not download models by
    name. Pass a local path to a Parakeet GGUF file (see
    ``ggml-org/parakeet-GGUF`` on Hugging Face).
    """

    def __init__(
        self,
        model_path: str,
        n_threads: Optional[int] = None,
        context_params: Optional[ParakeetContextParams] = None,
        **params,
    ):
        """
        :param model_path: Path to a Parakeet GGUF model file.
        :param n_threads: Threads for inference (default: min(4, CPU count)).
        :param context_params: Optional dict with ``use_gpu`` / ``gpu_device``.
        :param params: Extra full-params fields (e.g. ``offset_ms``, ``duration_ms``).
        """
        self._ctx = None
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Parakeet model not found: {model_path}. "
                "Download a GGUF from https://huggingface.co/ggml-org/parakeet-GGUF"
            )
        self.model_path = str(path.resolve())
        self._context_params = self._resolve_context_params(context_params)
        strategy = pw.parakeet_sampling_strategy.PARAKEET_SAMPLING_GREEDY
        self._params = pw.parakeet_full_default_params(strategy)
        if n_threads is None:
            n_threads = min(4, os.cpu_count() or 4)
        self._params.n_threads = int(n_threads)
        for key, value in params.items():
            if not hasattr(self._params, key):
                raise TypeError(f"Unknown Parakeet full param: {key}")
            setattr(self._params, key, value)
        self._init_model()

    def transcribe(
        self,
        media: Union[str, np.ndarray],
        **params,
    ) -> List[Segment]:
        """
        Transcribe a media path or float32 mono PCM at 16 kHz.

        :param media: File path (wav preferred; other formats need ffmpeg) or
                      1-D ``numpy.ndarray`` of float32 samples.
        :param params: Temporary overrides for full params (e.g. ``n_threads``).
        :return: List of :class:`~pywhispercpp.model.Segment` objects.
        """
        if isinstance(media, np.ndarray):
            audio = np.ascontiguousarray(media, dtype=np.float32).ravel()
        else:
            if not Path(media).exists():
                raise FileNotFoundError(media)
            audio = self._load_audio(str(media))

        for key, value in params.items():
            if not hasattr(self._params, key):
                raise TypeError(f"Unknown Parakeet full param: {key}")
            setattr(self._params, key, value)

        logger.info("Transcribing with Parakeet ...")
        ret = pw.parakeet_full(self._ctx, self._params, audio, int(audio.size))
        if ret != 0:
            raise RuntimeError(f"parakeet_full failed with code {ret}")

        n = pw.parakeet_full_n_segments(self._ctx)
        segments: List[Segment] = []
        for i in range(n):
            t0 = pw.parakeet_full_get_segment_t0(self._ctx, i)
            t1 = pw.parakeet_full_get_segment_t1(self._ctx, i)
            raw = pw.parakeet_full_get_segment_text(self._ctx, i)
            text = raw.decode("utf-8", errors="replace").strip()
            segments.append(Segment(t0, t1, text))
        return segments

    # Alias used by some call sites / docs
    transcribe_samples = transcribe

    def get_params(self) -> dict:
        res = {}
        for name in dir(self._params):
            if name.startswith("_"):
                continue
            try:
                res[name] = getattr(self._params, name)
            except Exception:
                continue
        return res

    @staticmethod
    def version() -> str:
        return pw.parakeet_version()

    @staticmethod
    def _resolve_context_params(context_params: Optional[ParakeetContextParams]):
        resolved = pw.parakeet_context_default_params()
        if context_params is None:
            return resolved
        if not isinstance(context_params, dict):
            raise TypeError("context_params must be a dict or None")
        unknown = sorted(set(context_params) - _CONTEXT_PARAM_KEYS)
        if unknown:
            raise TypeError(f"Unknown context_params keys: {', '.join(unknown)}")
        for key, value in context_params.items():
            setattr(resolved, key, value)
        return resolved

    def _init_model(self) -> None:
        logger.info("Initializing Parakeet model from %s ...", self.model_path)
        self._ctx = pw.parakeet_init_from_file_with_params(
            self.model_path, self._context_params
        )
        # pybind returns an opaque wrapper; a failed load leaves a null internal ptr.
        # Exercise free/n_segments carefully after a failed load.
        if self._ctx is None:
            raise RuntimeError(f"Failed to load Parakeet model: {self.model_path}")

    @staticmethod
    def _load_audio(media_file_path: str) -> np.ndarray:
        def wav_to_np(file_path: str) -> np.ndarray:
            with wave.open(file_path, "rb") as wf:
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                num_frames = wf.getnframes()

                if num_channels not in (1, 2):
                    raise Exception("WAV file must be mono or stereo")
                if sample_rate != pw.PARAKEET_SAMPLE_RATE:
                    raise Exception(f"WAV file must be {pw.PARAKEET_SAMPLE_RATE} Hz")
                if sample_width != 2:
                    raise Exception("WAV file must be 16-bit")

                raw = wf.readframes(num_frames)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                if num_channels == 1:
                    return audio / 32768.0
                audio = audio.reshape(-1, 2)
                return (audio[:, 0] + audio[:, 1]) / 65536.0

        if media_file_path.endswith(".wav"):
            return wav_to_np(media_file_path)

        if shutil.which("ffmpeg") is None:
            raise Exception(
                "FFMPEG is not installed or not in PATH. "
                "Install it, or provide a 16 kHz WAV or a NumPy array."
            )

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    media_file_path,
                    "-ac",
                    "1",
                    "-ar",
                    str(pw.PARAKEET_SAMPLE_RATE),
                    temp_file_path,
                    "-y",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return wav_to_np(temp_file_path)
        finally:
            os.remove(temp_file_path)

    def __del__(self):
        ctx = getattr(self, "_ctx", None)
        if ctx is not None:
            try:
                pw.parakeet_free(ctx)
            except Exception:
                pass
            self._ctx = None
