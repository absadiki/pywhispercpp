

import _pywhispercpp as pw

import unittest
from unittest import TestCase


class TestCAPI(TestCase):

    model_file = './whisper.cpp/models/for-tests-ggml-tiny.en.bin'

    def test_whisper_init_from_file(self):
        ctx = pw.whisper_init_from_file_with_params(
            self.model_file,
            pw.whisper_context_default_params(),
        )
        self.assertIsInstance(ctx, pw.whisper_context)

    def test_whisper_lang_str(self):
        return self.assertEqual(pw.whisper_lang_str(0), 'en')

    def test_whisper_lang_id(self):
        return self.assertEqual(pw.whisper_lang_id('en'), 0)

    def test_whisper_full_params_language_set_to_de(self):
        params = pw.whisper_full_params()
        params.language = 'de'
        return self.assertEqual(params.language, 'de')
    
    def test_whisper_full_params_language_set_to_german(self):
        params = pw.whisper_full_params()
        params.language = 'german'
        return self.assertEqual(params.language, 'de')
    
    def test_whisper_full_params_context(self):    
    
        params = pw.whisper_full_params()
        # to ensure that the string is not cached
        prompt = str(10120923) + "A" + " test"
        params.initial_prompt = prompt
        print("Params Prompt: ", params.initial_prompt)
        del prompt
        import gc
        gc.collect()
        return self.assertEqual(params.initial_prompt, str(10120923) + "A test")
    
    def test_whisper_full_params_regex(self):    
        params = pw.whisper_full_params()
        val = str(10120923) + "A" + " test"
        params.suppress_regex = val
        print("Params Prompt: ", params.suppress_regex)
        del val
        import gc
        gc.collect()
        return self.assertEqual(params.suppress_regex, str(10120923) + "A" + " test") 

    def test_whisper_full_params_default(self):
        params = pw.whisper_full_default_params(pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY)
        self.assertIsInstance(params, pw.whisper_full_params)
        self.assertEqual(params.suppress_regex, "")
    
    def test_whisper_lang_id(self):
        return self.assertEqual(pw.whisper_lang_id('en'), 0)
    
    def test_whisper_full_params(self):
        params = pw.whisper_full_params()
        return self.assertIsInstance(params.n_threads, int)


if __name__ == '__main__':
    
    unittest.main()
