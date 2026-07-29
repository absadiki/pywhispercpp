#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test parakeet_model.py
"""
import unittest
from pathlib import Path
from unittest import TestCase

import _pywhispercpp as pw
import numpy as np
import pywhispercpp.utils as utils
from pywhispercpp.base import Segment
from pywhispercpp.parakeet_model import (
    ParakeetModel,
    ParakeetParams,
    ParakeetContextParams,
)

WHISPER_CPP_DIR = Path(__file__).parent.parent / 'whisper.cpp'


class TestParakeetParams(TestCase):
    """Tests for ParakeetParams dataclass."""

    def test_default_values(self):
        params = ParakeetParams()
        self.assertIsNone(params.n_threads)
        self.assertIsNone(params.offset_ms)
        self.assertIsNone(params.duration_ms)
        self.assertIsNone(params.no_context)
        self.assertIsNone(params.audio_ctx)

    def test_custom_values(self):
        params = ParakeetParams(
            n_threads=8, offset_ms=100, duration_ms=5000,
            no_context=True, audio_ctx=256,
        )
        self.assertEqual(params.n_threads, 8)
        self.assertEqual(params.offset_ms, 100)
        self.assertEqual(params.duration_ms, 5000)
        self.assertTrue(params.no_context)
        self.assertEqual(params.audio_ctx, 256)

    def test_apply_to_c_params(self):
        c_params = pw.parakeet_full_default_params(
            pw.parakeet_sampling_strategy.PARAKEET_SAMPLING_GREEDY
        )
        params = ParakeetParams(n_threads=16)
        params.apply(c_params)
        self.assertEqual(c_params.n_threads, 16)

    def test_apply_partial_params(self):
        c_params = pw.parakeet_full_default_params(
            pw.parakeet_sampling_strategy.PARAKEET_SAMPLING_GREEDY
        )
        params = ParakeetParams(no_context=True)
        params.apply(c_params)
        self.assertTrue(c_params.no_context)

    def test_apply_ignores_none_values(self):
        c_params = pw.parakeet_full_default_params(
            pw.parakeet_sampling_strategy.PARAKEET_SAMPLING_GREEDY
        )
        original_n_threads = c_params.n_threads
        params = ParakeetParams(n_threads=None, offset_ms=None)
        params.apply(c_params)
        self.assertEqual(c_params.n_threads, original_n_threads)


class TestParakeetContextParams(TestCase):
    """Tests for ParakeetContextParams dataclass."""

    def test_default_values(self):
        params = ParakeetContextParams()
        self.assertIsNone(params.use_gpu)
        self.assertIsNone(params.gpu_device)

    def test_custom_values(self):
        params = ParakeetContextParams(use_gpu=True, gpu_device=0)
        self.assertTrue(params.use_gpu)
        self.assertEqual(params.gpu_device, 0)

    def test_apply_to_c_params(self):
        c_params = pw.parakeet_context_default_params()
        params = ParakeetContextParams(use_gpu=False)
        params.apply(c_params)
        self.assertFalse(c_params.use_gpu)

    def test_apply_ignores_none_values(self):
        c_params = pw.parakeet_context_default_params()
        original_use_gpu = c_params.use_gpu
        params = ParakeetContextParams(use_gpu=None)
        params.apply(c_params)
        self.assertEqual(c_params.use_gpu, original_use_gpu)


class TestParakeetModel(TestCase):
    audio_file = WHISPER_CPP_DIR / 'samples/jfk.wav'
    model = ParakeetModel(
        str(WHISPER_CPP_DIR / 'models/for-tests-ggml-parakeet-tdt.bin'),
        redirect_logs_to=None,
    )

    def test_transcribe_returns_list(self):
        segments = self.model.transcribe(str(self.audio_file))
        self.assertIsInstance(segments, list)

    def test_transcribe_numpy_array(self):
        audio = utils.load_audio(str(self.audio_file), sample_rate=pw.PARAKEET_SAMPLE_RATE)
        segments = self.model.transcribe(audio)
        self.assertIsInstance(segments, list)

    def test_transcribe_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.model.transcribe('/nonexistent/file.wav')

    def test_transcribe_with_params(self):
        params = ParakeetParams(n_threads=4)
        segments = self.model.transcribe(str(self.audio_file), params=params)
        self.assertIsInstance(segments, list)

    def test_transcribe_with_kwargs(self):
        segments = self.model.transcribe(str(self.audio_file), n_threads=4)
        self.assertIsInstance(segments, list)

    def test_transcribe_with_new_segment_callback(self):
        callback_segments = []
        self.model.transcribe(
            str(self.audio_file),
            new_segment_callback=lambda seg: callback_segments.append(seg)
        )
        self.assertIsInstance(callback_segments, list)

    def test_transcribe_with_abort_callback(self):
        segments = self.model.transcribe(
            str(self.audio_file),
            abort_callback=lambda: False
        )
        self.assertIsInstance(segments, list)

    def test_model_init_with_params(self):
        params = ParakeetParams(n_threads=2)
        model = ParakeetModel(
            str(WHISPER_CPP_DIR / 'models/for-tests-ggml-parakeet-tdt.bin'),
            params=params,
            redirect_logs_to=None,
        )
        self.assertIsNotNone(model._ctx)

    def test_model_init_with_kwargs(self):
        model = ParakeetModel(
            str(WHISPER_CPP_DIR / 'models/for-tests-ggml-parakeet-tdt.bin'),
            n_threads=2,
            redirect_logs_to=None,
        )
        self.assertIsNotNone(model._ctx)

    def test_model_init_with_context_params(self):
        context_params = ParakeetContextParams(use_gpu=False)
        model = ParakeetModel(
            str(WHISPER_CPP_DIR / 'models/for-tests-ggml-parakeet-tdt.bin'),
            context_params=context_params,
            redirect_logs_to=None,
        )
        self.assertIsNotNone(model._ctx)

    def test_print_timings(self):
        self.model.transcribe(str(self.audio_file))
        self.model.print_timings()

    def test_print_system_info(self):
        self.model.print_system_info()

    def test_model_metadata(self):
        ctx = self.model._ctx
        self.assertGreater(pw.parakeet_model_n_vocab(ctx), 0)
        self.assertGreater(pw.parakeet_model_n_audio_ctx(ctx), 0)
        self.assertGreater(pw.parakeet_model_n_audio_state(ctx), 0)
        self.assertGreater(pw.parakeet_model_n_audio_head(ctx), 0)
        self.assertGreater(pw.parakeet_model_n_audio_layer(ctx), 0)
        self.assertGreater(pw.parakeet_model_n_mels(ctx), 0)


class TestParakeetCAPI(TestCase):
    """Tests for low-level parakeet C bindings."""

    model_file = str(WHISPER_CPP_DIR / 'models/for-tests-ggml-parakeet-tdt.bin')

    def test_parakeet_version(self):
        version = pw.parakeet_version()
        self.assertIsInstance(version, str)
        self.assertGreater(len(version), 0)

    def test_parakeet_context_default_params(self):
        params = pw.parakeet_context_default_params()
        self.assertTrue(hasattr(params, 'use_gpu'))
        self.assertTrue(hasattr(params, 'gpu_device'))

    def test_parakeet_full_default_params(self):
        params = pw.parakeet_full_default_params(
            pw.parakeet_sampling_strategy.PARAKEET_SAMPLING_GREEDY
        )
        self.assertIsNotNone(params)

    def test_parakeet_init_from_file(self):
        ctx = pw.parakeet_init_from_file_with_params(
            self.model_file,
            pw.parakeet_context_default_params(),
        )
        self.assertIsNotNone(ctx)
        pw.parakeet_free(ctx)

    def test_parakeet_n_vocab(self):
        ctx = pw.parakeet_init_from_file_with_params(
            self.model_file,
            pw.parakeet_context_default_params(),
        )
        self.assertGreater(pw.parakeet_n_vocab(ctx), 0)
        pw.parakeet_free(ctx)

    def test_parakeet_token_blank(self):
        ctx = pw.parakeet_init_from_file_with_params(
            self.model_file,
            pw.parakeet_context_default_params(),
        )
        self.assertIsInstance(pw.parakeet_token_blank(ctx), int)
        pw.parakeet_free(ctx)


if __name__ == '__main__':
    unittest.main()
