#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test model.py
"""
import unittest
from pathlib import Path
from unittest import TestCase

import _pywhispercpp as pw
from pywhispercpp.model import Model, Segment

if __name__ == '__main__':
    pass

WHISPER_CPP_DIR = Path(__file__).parent.parent / 'whisper.cpp'

class TestModel(TestCase):
    audio_file = WHISPER_CPP_DIR/ 'samples/jfk.wav'
    model = Model("tiny", models_dir=str(WHISPER_CPP_DIR/'models'))

    def test_transcribe(self):
        segments = self.model.transcribe(str(self.audio_file))
        return self.assertIsInstance(segments, list) and \
               self.assertIsInstance(segments[0], Segment) if len(segments) > 0 else True

    def test_get_params(self):
        params = self.model.get_params()
        return self.assertIsInstance(params, dict)

    def test_lang_max_id(self):
        n = self.model.lang_max_id()
        return self.assertGreater(n, 0)

    def test_available_languages(self):
        av_langs = self.model.available_languages()
        return self.assertIsInstance(av_langs, list) and self.assertGreater(len(av_langs), 1)

    def test__load_audio(self):
        audio_arr = self.model._load_audio(str(self.audio_file))
        return self.assertIsNotNone(audio_arr)

    def test_auto_detect_language(self):
        detected_language, probs = self.model.auto_detect_language(str(self.audio_file))
        return self.assertIsInstance(detected_language, tuple) and self.assertEqual(detected_language[0], 'en')

    def test_context_params_dict_init(self):
        model = Model(
            "tiny",
            models_dir=str(WHISPER_CPP_DIR/'models'),
            context_params={"use_gpu": False, "flash_attn": False},
        )
        self.assertIsInstance(model, Model)

    def test_compat_alias_for_non_speech_tokens(self):
        model = Model(
            "tiny",
            models_dir=str(WHISPER_CPP_DIR/'models'),
            suppress_non_speech_tokens=True,
        )
        self.assertTrue(model.get_params()["suppress_nst"])

    def test_prompt_token_helper_exists(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )
        params.set_prompt_tokens((1, 2, 3))
        self.assertEqual(params.prompt_n_tokens, 3)

    def test_model_metadata_bindings(self):
        self.assertIsInstance(pw.whisper_model_type_readable(self.model._ctx), str)
        self.assertGreater(pw.whisper_model_n_vocab(self.model._ctx), 0)
        self.assertGreater(pw.whisper_model_n_audio_ctx(self.model._ctx), 0)
        self.assertGreater(pw.whisper_model_n_text_ctx(self.model._ctx), 0)

    def test_speaker_turn_accessor_smoke(self):
        self.model.transcribe(str(self.audio_file))
        segment_count = pw.whisper_full_n_segments(self.model._ctx)
        self.assertGreater(segment_count, 0)
        self.assertIsInstance(
            pw.whisper_full_get_segment_speaker_turn_next(self.model._ctx, 0),
            bool,
        )


if __name__ == '__main__':
    unittest.main()
