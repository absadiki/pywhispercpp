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
from typing import Union, Callable, List, TextIO, Tuple
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


class Segment:
    """
    A small class representing a transcription segment
    """

    def __init__(self, t0: int, t1: int, text: str):
        """
        :param t0: start time
        :param t1: end time
        :param text: text
        """
        self.t0 = t0
        self.t1 = t1
        self.text = text

    def __str__(self):
        return f"t0={self.t0}, t1={self.t1}, text={self.text}"

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
                 models_dir: str = None,
                 params_sampling_strategy: int = 0,
                 redirect_whispercpp_logs_to: Union[bool, TextIO, str, None] = False,
                 use_openvino: bool = False,
                 openvino_model_path: str = None,
                 openvino_device: str = 'CPU',
                 openvino_cache_dir: str = None,
                 **params):
        """
        :param model: The name of the model, one of the [AVAILABLE_MODELS](/pywhispercpp/#pywhispercpp.constants.AVAILABLE_MODELS),
                        (default to `tiny`), or a direct path to a `ggml` model.
        :param models_dir: The directory where the models are stored, or where they will be downloaded if they don't
                            exist, default to [MODELS_DIR](/pywhispercpp/#pywhispercpp.constants.MODELS_DIR) <user_data_dir/pywhsipercpp/models>
        :param params_sampling_strategy: 0 -> GREEDY, else BEAM_SEARCH
        :param redirect_whispercpp_logs_to: where to redirect the whisper.cpp logs, default to False (no redirection), accepts str file path, sys.stdout, sys.stderr, or use None to redirect to devnull
        :param use_openvino: whether to use OpenVINO or not
        :param openvino_model_path: path to the OpenVINO model
        :param openvino_device: OpenVINO device, default to CPU
        :param openvino_cache_dir: OpenVINO cache directory
        :param params: keyword arguments for different whisper.cpp parameters,
                        see [PARAMS_SCHEMA](/pywhispercpp/#pywhispercpp.constants.PARAMS_SCHEMA)
        """
        if Path(model).is_file():
            self.model_path = model
        else:
            self.model_path = utils.download_model(model, models_dir)
        self._ctx = None
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
                   n_processors: int = None,
                   new_segment_callback: Callable[[Segment], None] = None,
                   **params) -> List[Segment]:
        """
        Transcribes the media provided as input and returns list of `Segment` objects.
        Accepts a media_file path (audio/video) or a raw numpy array.

        :param media: Media file path or a numpy array
        :param n_processors: if not None, it will run the transcription on multiple processes
                             binding to whisper.cpp/whisper_full_parallel
                             > Split the input audio in chunks and process each chunk separately using whisper_full()
        :param new_segment_callback: callback function that will be called when a new segment is generated
        :param params: keyword arguments for different whisper.cpp parameters, see ::: constants.PARAMS_SCHEMA

        :return: List of transcription segments
        """
        if type(media) is np.ndarray:
            audio = media
        else:
            if not Path(media).exists():
                raise FileNotFoundError(media)
            audio = self._load_audio(media)
        # update params if any
        self._set_params(params)

        # setting up callback
        if new_segment_callback:
            Model._new_segment_callback = new_segment_callback
            pw.assign_new_segment_callback(self._params, Model.__call_new_segment_callback)

        # run inference
        start_time = time()
        logger.info("Transcribing ...")
        res = self._transcribe(audio, n_processors=n_processors)
        end_time = time()
        logger.info(f"Inference time: {end_time - start_time:.3f} s")
        return res

    @staticmethod
    def _get_segments(ctx, start: int, end: int) -> List[Segment]:
        """
        Helper function to get generated segments between `start` and `end`

        :param start: start index
        :param end: end index

        :return: list of segments
        """
        n = pw.whisper_full_n_segments(ctx)
        assert end <= n, f"{end} > {n}: `End` index must be less or equal than the total number of segments"
        res = []
        for i in range(start, end):
            t0 = pw.whisper_full_get_segment_t0(ctx, i)
            t1 = pw.whisper_full_get_segment_t1(ctx, i)
            bytes = pw.whisper_full_get_segment_text(ctx, i)
            text = bytes.decode('utf-8',errors='replace')
            res.append(Segment(t0, t1, text.strip()))
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
        Returns number of supported languages.
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
    def available_languages() -> list[str]:
        """
        Returns a list of supported language codes

        :return: list of supported language codes
        """
        n = pw.whisper_lang_max_id()
        res = []
        for i in range(n):
            res.append(pw.whisper_lang_str(i))
        return res

    def _init_model(self) -> None:
        """
        Private method to initialize the method from the bindings, it will be called automatically from the __init__
        :return:
        """
        logger.info("Initializing the model ...")
        with utils.redirect_stderr(to=self.redirect_whispercpp_logs_to):
            self._ctx = pw.whisper_init_from_file(self.model_path)
            if self.use_openvino:
                pw.whisper_ctx_init_openvino_encoder(self._ctx, self.openvino_model_path, self.openvino_device, self.openvino_cache_dir)



    def _set_params(self, kwargs: dict) -> None:
        """
        Private method to set the kwargs params to the `Params` class
        :param kwargs: dict like object for the different params
        :return: None
        """
        for param in kwargs:
            setattr(self._params, param, kwargs[param])

    def _transcribe(self, audio: np.ndarray, n_processors: int = None):
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
        res = Model._get_segments(self._ctx, 0, n)
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
        res = Model._get_segments(ctx, start, n)
        for segment in res:
            Model._new_segment_callback(segment)

    @staticmethod
    def _load_audio(media_file_path: str) -> np.array:
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

    def auto_detect_language(self,  media: Union[str, np.ndarray], offset_ms: int = 0, n_threads: int = 4) -> Tuple[Tuple[str, np.float32], dict[str, np.float32]]:
        """
        Automatic language detection using whisper.cpp/whisper_pcm_to_mel and whisper.cpp/whisper_lang_auto_detect

        :param media: Media file path or a numpy array
        :param offset_ms: offset in milliseconds
        :param n_threads: number of threads to use
        :return: ((detected_language, probability), probabilities for all languages)
        """
        if type(media) is np.ndarray:
            audio = media
        else:
            if not Path(media).exists():
                raise FileNotFoundError(media)
            audio = self._load_audio(media)

        pw.whisper_pcm_to_mel(self._ctx, audio, len(audio), n_threads)
        lang_max_id = self.lang_max_id()
        probs = np.zeros(lang_max_id, dtype=np.float32)
        auto_detect = pw.whisper_lang_auto_detect(self._ctx, offset_ms, n_threads, probs)
        langs = self.available_languages()
        lang_probs = {langs[i]: probs[i] for i in range(lang_max_id)}
        return (langs[auto_detect], probs[auto_detect]), lang_probs

    def __del__(self):
        """
        Free up resources
        :return: None
        """
        pw.whisper_free(self._ctx)