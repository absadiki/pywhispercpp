#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest import TestCase

import _pywhispercpp as pw

from pywhispercpp.model import Model, Segment


WHISPER_CPP_DIR = Path(__file__).parent.parent / 'whisper.cpp'


class TestBackwardsCompatibility(TestCase):
    audio_file = WHISPER_CPP_DIR / 'samples/jfk.wav'
    models_dir = str(WHISPER_CPP_DIR / 'models')
    repo_root = Path(__file__).parent.parent

    def tearDown(self):
        gc.collect()

    def _create_cpu_model(self):
        return Model(
            'tiny',
            models_dir=self.models_dir,
            context_params={'use_gpu': False, 'flash_attn': False},
        )

    def _run_python(self, code: str):
        result = subprocess.run(
            [sys.executable, '-c', textwrap.dedent(code)],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )

    def test_legacy_model_constructor_still_works(self):
        self._run_python(
            f'''
            from pywhispercpp.model import Model

            model = Model('tiny', models_dir={self.models_dir!r})
            assert isinstance(model, Model)
            '''
        )

    def test_legacy_alias_still_maps_to_suppress_nst(self):
        self._run_python(
            f'''
            from pywhispercpp.model import Model

            model = Model(
                'tiny',
                models_dir={self.models_dir!r},
                context_params={{'use_gpu': False, 'flash_attn': False}},
            )
            model._set_params({{'suppress_non_speech_tokens': True}})
            assert model.get_params()['suppress_nst'] is True
            '''
        )

    def test_low_level_prompt_tokens_property_round_trips(self):
        params = pw.whisper_full_default_params(
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        )
        params.prompt_tokens = (1, 2, 3)
        self.assertEqual(tuple(params.prompt_tokens), (1, 2, 3))
        self.assertEqual(params.prompt_n_tokens, 3)

    def test_context_params_dict_is_additive(self):
        self._run_python(
            f'''
            from pywhispercpp.model import Model

            model = Model(
                'tiny',
                models_dir={self.models_dir!r},
                context_params={{'use_gpu': False, 'flash_attn': False}},
            )
            assert isinstance(model, Model)
            '''
        )

    def test_existing_new_segment_callback_still_works(self):
        self._run_python(
            f'''
            from pywhispercpp.model import Model, Segment

            seen = []
            model = Model(
                'tiny',
                models_dir={self.models_dir!r},
                context_params={{'use_gpu': False, 'flash_attn': False}},
            )

            def on_segment(segment):
                seen.append(segment)

            segments = model.transcribe({str(self.audio_file)!r}, new_segment_callback=on_segment)
            assert isinstance(segments, list)
            assert len(seen) > 0
            assert all(isinstance(segment, Segment) for segment in seen)
            '''
        )

    def test_abort_callback_can_abort_and_then_clear(self):
        self._run_python(
            f'''
            from pywhispercpp.model import Model

            model = Model(
                'tiny',
                models_dir={self.models_dir!r},
                context_params={{'use_gpu': False, 'flash_attn': False}},
            )
            callback_calls = []

            def abort_immediately():
                callback_calls.append(True)
                return True

            aborted_segments = model.transcribe({str(self.audio_file)!r}, abort_callback=abort_immediately)
            assert isinstance(aborted_segments, list)
            assert len(callback_calls) > 0

            normal_segments = model.transcribe({str(self.audio_file)!r})
            assert isinstance(normal_segments, list)
            assert len(normal_segments) > 0
            '''
        )

    def test_log_callback_can_be_set_and_cleared(self):
        pw.whisper_log_set(lambda level, text: None)
        pw.whisper_log_set(None)

    def test_alignment_preset_enum_is_available(self):
        preset = pw.whisper_alignment_heads_preset.WHISPER_AHEADS_TINY
        self.assertIsNotNone(preset)


if __name__ == '__main__':
    unittest.main()