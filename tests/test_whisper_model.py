#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test whisper_model.py
"""
import unittest
import warnings
from pathlib import Path
from unittest import TestCase

import _pywhispercpp as pw
import numpy as np
import pywhispercpp.utils as utils
from pywhispercpp.base import Segment
from pywhispercpp.whisper_model import (
    WhisperModel,
    WhisperParams,
    WhisperContextParams,
)

WHISPER_CPP_DIR = Path(__file__).parent.parent / 'whisper.cpp'


class TestWhisperParams(TestCase):
    """Tests for WhisperParams dataclass."""

    def test_default_values(self):
        params = WhisperParams()
        self.assertIsNone(params.n_threads)
        self.assertIsNone(params.translate)
        self.assertIsNone(params.language)
        self.assertIsNone(params.temperature)

    def test_custom_values(self):
        params = WhisperParams(
            n_threads=8, translate=True, language='de',
            temperature=0.5, no_context=False,
        )
        self.assertEqual(params.n_threads, 8)
        self.assertTrue(params.translate)
        self.assertEqual(params.language, 'de')
        self.assertEqual(params.temperature, 0.5)
        self.assertFalse(params.no_context)

    def test_apply_to_c_params(self):
        c_params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )
        params = WhisperParams(n_threads=16, language='fr')
        params.apply(c_params)
        self.assertEqual(c_params.n_threads, 16)
        self.assertEqual(c_params.language, 'fr')

    def test_apply_partial_params(self):
        c_params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )
        params = WhisperParams(no_context=True)
        params.apply(c_params)
        self.assertTrue(c_params.no_context)

    def test_apply_ignores_none_values(self):
        c_params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )
        original_n_threads = c_params.n_threads
        params = WhisperParams(n_threads=None)
        params.apply(c_params)
        self.assertEqual(c_params.n_threads, original_n_threads)

    def test_suppress_non_speech_tokens_alias(self):
        c_params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )
        params = WhisperParams(suppress_non_speech_tokens=True)
        params.apply(c_params)
        self.assertTrue(c_params.suppress_nst)

    def test_prompt_tokens(self):
        c_params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )
        params = WhisperParams(prompt_tokens=(1, 2, 3))
        params.apply(c_params)
        self.assertEqual(c_params.prompt_n_tokens, 3)


class TestWhisperContextParams(TestCase):
    """Tests for WhisperContextParams dataclass."""

    def test_default_values(self):
        params = WhisperContextParams()
        self.assertIsNone(params.use_gpu)
        self.assertIsNone(params.flash_attn)
        self.assertIsNone(params.gpu_device)

    def test_custom_values(self):
        params = WhisperContextParams(use_gpu=True, flash_attn=False, gpu_device=1)
        self.assertTrue(params.use_gpu)
        self.assertFalse(params.flash_attn)
        self.assertEqual(params.gpu_device, 1)

    def test_apply_to_c_params(self):
        c_params = pw.whisper_context_default_params()
        params = WhisperContextParams(use_gpu=False)
        params.apply(c_params)
        self.assertFalse(c_params.use_gpu)

    def test_apply_ignores_none_values(self):
        c_params = pw.whisper_context_default_params()
        original_use_gpu = c_params.use_gpu
        params = WhisperContextParams(use_gpu=None)
        params.apply(c_params)
        self.assertEqual(c_params.use_gpu, original_use_gpu)


class TestWhisperModel(TestCase):
    audio_file = WHISPER_CPP_DIR / 'samples/jfk.wav'
    model = WhisperModel("tiny", models_dir=str(WHISPER_CPP_DIR / 'models'), redirect_logs_to=None)

    def test_transcribe_returns_list(self):
        segments = self.model.transcribe(str(self.audio_file))
        self.assertIsInstance(segments, list)

    def test_transcribe_returns_segments(self):
        segments = self.model.transcribe(str(self.audio_file))
        if len(segments) > 0:
            self.assertIsInstance(segments[0], Segment)

    def test_transcribe_jfk_content(self):
        segments = self.model.transcribe(str(self.audio_file))
        full_text = ' '.join(seg.text for seg in segments).lower()
        self.assertIn('fellow americans', full_text)

    def test_transcribe_numpy_array(self):
        audio = utils.load_audio(str(self.audio_file), sample_rate=pw.WHISPER_SAMPLE_RATE)
        segments = self.model.transcribe(audio)
        self.assertIsInstance(segments, list)
        if len(segments) > 0:
            self.assertIsInstance(segments[0], Segment)

    def test_transcribe_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.model.transcribe('/nonexistent/file.wav')

    def test_transcribe_with_params(self):
        params = WhisperParams(n_threads=4)
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
        self.assertGreater(len(callback_segments), 0)
        self.assertIsInstance(callback_segments[0], Segment)

    def test_transcribe_with_abort_callback(self):
        segments = self.model.transcribe(
            str(self.audio_file),
            abort_callback=lambda: False
        )
        self.assertIsInstance(segments, list)

    def test_transcribe_with_extract_probability(self):
        segments = self.model.transcribe(
            str(self.audio_file),
            extract_probability=True
        )
        self.assertIsInstance(segments, list)
        if len(segments) > 0:
            self.assertFalse(np.isnan(segments[0].probability))

    def test_model_init_with_params(self):
        params = WhisperParams(n_threads=2, language='en')
        model = WhisperModel(
            "tiny",
            models_dir=str(WHISPER_CPP_DIR / 'models'),
            params=params,
            redirect_logs_to=None,
        )
        self.assertIsNotNone(model._ctx)

    def test_model_init_with_kwargs(self):
        model = WhisperModel(
            "tiny",
            models_dir=str(WHISPER_CPP_DIR / 'models'),
            n_threads=2,
            redirect_logs_to=None,
        )
        self.assertIsNotNone(model._ctx)

    def test_model_init_with_context_params(self):
        context_params = WhisperContextParams(use_gpu=False)
        model = WhisperModel(
            "tiny",
            models_dir=str(WHISPER_CPP_DIR / 'models'),
            context_params=context_params,
            redirect_logs_to=None,
        )
        self.assertIsNotNone(model._ctx)

    def test_model_init_beam_search(self):
        model = WhisperModel(
            "tiny",
            models_dir=str(WHISPER_CPP_DIR / 'models'),
            sampling_strategy=1,
            redirect_logs_to=None,
        )
        self.assertIsNotNone(model._ctx)

    def test_print_timings(self):
        self.model.transcribe(str(self.audio_file))
        self.model.print_timings()

    def test_print_system_info(self):
        self.model.print_system_info()

    def test_lang_max_id(self):
        n = WhisperModel.lang_max_id()
        self.assertGreater(n, 0)

    def test_available_languages(self):
        langs = WhisperModel.available_languages()
        self.assertIsInstance(langs, list)
        self.assertGreater(len(langs), 1)
        self.assertIn('en', langs)

    def test_auto_detect_language(self):
        detected_language, probs = self.model.auto_detect_language(str(self.audio_file))
        self.assertIsInstance(detected_language, tuple)
        self.assertEqual(detected_language[0], 'en')

    def test_auto_detect_language_numpy(self):
        audio = utils.load_audio(str(self.audio_file), sample_rate=pw.WHISPER_SAMPLE_RATE)
        detected_language, probs = self.model.auto_detect_language(audio)
        self.assertEqual(detected_language[0], 'en')

    def test_model_metadata(self):
        ctx = self.model._ctx
        self.assertIsInstance(pw.whisper_model_type_readable(ctx), str)
        self.assertGreater(pw.whisper_model_n_vocab(ctx), 0)
        self.assertGreater(pw.whisper_model_n_audio_ctx(ctx), 0)
        self.assertGreater(pw.whisper_model_n_text_ctx(ctx), 0)

    def test_speaker_turn_accessor(self):
        self.model.transcribe(str(self.audio_file))
        segment_count = pw.whisper_full_n_segments(self.model._ctx)
        self.assertGreater(segment_count, 0)
        self.assertIsInstance(
            pw.whisper_full_get_segment_speaker_turn_next(self.model._ctx, 0),
            bool,
        )

    def test_compat_alias_for_non_speech_tokens(self):
        model = WhisperModel(
            "tiny",
            models_dir=str(WHISPER_CPP_DIR / 'models'),
            suppress_non_speech_tokens=True,
            redirect_logs_to=None,
        )
        self.assertTrue(model._params.suppress_nst)


class TestWhisperCAPI(TestCase):
    """Tests for low-level whisper C bindings."""

    model_file = str(WHISPER_CPP_DIR / 'models/for-tests-ggml-tiny.en.bin')

    def test_whisper_init_from_file(self):
        ctx = pw.whisper_init_from_file_with_params(
            self.model_file,
            pw.whisper_context_default_params(),
        )
        self.assertIsInstance(ctx, pw.whisper_context)
        pw.whisper_free(ctx)

    def test_whisper_lang_str(self):
        self.assertEqual(pw.whisper_lang_str(0), 'en')

    def test_whisper_lang_id(self):
        self.assertEqual(pw.whisper_lang_id('en'), 0)

    def test_whisper_full_default_params(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )
        self.assertIsInstance(params, pw.whisper_full_params)
        self.assertEqual(params.suppress_regex, "")

    def test_whisper_full_params_language_set(self):
        params = pw.whisper_full_params()
        params.language = 'de'
        self.assertEqual(params.language, 'de')

    def test_whisper_full_params_prompt_tokens(self):
        params = pw.whisper_full_params()
        params.set_prompt_tokens((1, 2, 3))
        self.assertEqual(params.prompt_n_tokens, 3)


class TestDeprecationWarning(TestCase):
    """Tests for deprecation warning on old Model class."""

    def test_old_model_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from pywhispercpp.model import Model
            Model("tiny", models_dir=str(WHISPER_CPP_DIR / 'models'))
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertGreater(len(deprecation_warnings), 0)
            self.assertIn("WhisperModel", str(deprecation_warnings[0].message))


if __name__ == '__main__':
    unittest.main()
