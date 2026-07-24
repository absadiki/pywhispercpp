#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parakeet binding smoke tests (no large model download)."""

import unittest
from unittest import TestCase

import _pywhispercpp as pw


class TestParakeetBindings(TestCase):
    def test_import_and_constants(self):
        self.assertEqual(pw.PARAKEET_SAMPLE_RATE, 16000)

    def test_parakeet_version(self):
        version = pw.parakeet_version()
        self.assertIsInstance(version, str)
        self.assertGreater(len(version), 0)

    def test_context_default_params(self):
        params = pw.parakeet_context_default_params()
        params.use_gpu = False
        params.gpu_device = 0
        self.assertFalse(params.use_gpu)
        self.assertEqual(params.gpu_device, 0)

    def test_full_default_params(self):
        params = pw.parakeet_full_default_params(
            pw.parakeet_sampling_strategy.PARAKEET_SAMPLING_GREEDY
        )
        self.assertGreater(params.n_threads, 0)
        params.n_threads = 2
        self.assertEqual(params.n_threads, 2)

    def test_init_missing_file_returns_none(self):
        params = pw.parakeet_context_default_params()
        params.use_gpu = False
        ctx = pw.parakeet_init_from_file_with_params(
            "/nonexistent/path/parakeet-model.bin",
            params,
        )
        self.assertIsNone(ctx)

    def test_free_none_safe(self):
        pw.parakeet_free(None)

    def test_python_package_export(self):
        from pywhispercpp import ParakeetModel, Segment, Model

        self.assertTrue(callable(ParakeetModel))
        self.assertTrue(callable(Model))
        self.assertTrue(callable(Segment))

    def test_parakeet_model_missing_path(self):
        from pywhispercpp import ParakeetModel

        with self.assertRaises(FileNotFoundError):
            ParakeetModel("/no/such/parakeet.gguf")


if __name__ == "__main__":
    unittest.main()
