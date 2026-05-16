#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Constants
"""
from pathlib import Path
from typing import Tuple

import _pywhispercpp as _pwcpp
from platformdirs import user_data_dir


WHISPER_SAMPLE_RATE = _pwcpp.WHISPER_SAMPLE_RATE
# MODELS URL MODELS_BASE_URL+ '/' + MODELS_PREFIX_URL+'-'+MODEL_NAME+'.bin'
# example = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
MODELS_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp"
MODELS_PREFIX_URL = "resolve/main/ggml"


PACKAGE_NAME = 'pywhispercpp'


MODELS_DIR = Path(user_data_dir(PACKAGE_NAME)) / 'models'


AVAILABLE_MODELS = [
                "base",
                "base-q5_1",
                "base-q8_0",
                "base.en",
                "base.en-q5_1",
                "base.en-q8_0",
                "large-v1",
                "large-v2",
                "large-v2-q5_0",
                "large-v2-q8_0",
                "large-v3",
                "large-v3-q5_0",
                "large-v3-turbo",
                "large-v3-turbo-q5_0",
                "large-v3-turbo-q8_0",
                "medium",
                "medium-q5_0",
                "medium-q8_0",
                "medium.en",
                "medium.en-q5_0",
                "medium.en-q8_0",
                "small",
                "small-q5_1",
                "small-q8_0",
                "small.en",
                "small.en-q5_1",
                "small.en-q8_0",
                "tiny",
                "tiny-q5_1",
                "tiny-q8_0",
                "tiny.en",
                "tiny.en-q5_1",
                "tiny.en-q8_0",
                ]
PARAMS_SCHEMA = {  # as exactly presented in whisper.cpp
    'n_threads': {
            'type': int,
            'description': "Number of threads to allocate for the inference"
                           "default to min(4, available hardware_concurrency)",
            'options': None,
            'default': None
    },
    'n_max_text_ctx': {
            'type': int,
            'description': "max tokens to use from past text as prompt for the decoder",
            'options': None,
            'default': 16384
    },
    'offset_ms': {
            'type': int,
            'description': "start offset in ms",
            'options': None,
            'default': 0
    },
    'duration_ms': {
            'type': int,
            'description': "audio duration to process in ms",
            'options': None,
            'default': 0
    },
    'translate': {
            'type': bool,
            'description': "whether to translate the audio to English",
            'options': None,
            'default': False
    },
    'no_context': {
            'type': bool,
            'description': "do not use past transcription (if any) as initial prompt for the decoder",
            'options': None,
            'default': True
    },
    'no_timestamps': {
            'type': bool,
            'description': "do not generate timestamps",
            'options': None,
            'default': False
    },
    'single_segment': {
            'type': bool,
            'description': "force single segment output (useful for streaming)",
            'options': None,
            'default': False
    },
    'print_special': {
            'type': bool,
            'description': "print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)",
            'options': None,
            'default': False
    },
    'print_progress': {
            'type': bool,
            'description': "print progress information",
            'options': None,
            'default': True
    },
    'print_realtime': {
            'type': bool,
            'description': "print results from within whisper.cpp (avoid it, use callback instead)",
            'options': None,
            'default': False
    },
    'print_timestamps': {
            'type': bool,
            'description': "print timestamps for each text segment when printing realtime",
            'options': None,
            'default': True
    },
    # [EXPERIMENTAL] token-level timestamps
    'token_timestamps': {
            'type': bool,
            'description': "enable token-level timestamps",
            'options': None,
            'default': False
    },
    'thold_pt': {
            'type': float,
            'description': "timestamp token probability threshold (~0.01)",
            'options': None,
            'default': 0.01
    },
    'thold_ptsum': {
            'type': float,
            'description': "timestamp token sum probability threshold (~0.01)",
            'options': None,
            'default': 0.01
    },
    'max_len': {
            'type': int,
            'description': "max segment length in characters, note: token_timestamps needs to be set to True for this to work",
            'options': None,
            'default': 0
    },
    'split_on_word': {
            'type': bool,
            'description': "split on word rather than on token (when used with max_len)",
            'options': None,
            'default': False
    },
    'max_tokens': {
            'type': int,
            'description': "max tokens per segment (0 = no limit)",
            'options': None,
            'default': 0
    },
    'debug_mode': {
            'type': bool,
            'description': "enable debug mode in whisper.cpp",
            'options': None,
            'default': False
    },
    'audio_ctx': {
            'type': int,
            'description': "overwrite the audio context size (0 = use default)",
            'options': None,
            'default': 0
    },
    'tdrz_enable': {
            'type': bool,
            'description': "enable tinydiarize speaker turn detection",
            'options': None,
            'default': False
    },
    'initial_prompt': {
            'type': str,
            'description': "Initial prompt, these are prepended to any existing text context from a previous call",
            'options': None,
            'default': None
    },
    'grammar': {
            'type': str,
            'description': "GBNF grammar text or a path to a grammar file",
            'options': None,
            'default': None
    },
    'grammar_rule': {
            'type': str,
            'description': "top-level GBNF grammar rule name",
            'options': None,
            'default': 'root'
    },
    'prompt_tokens': {
            'type': Tuple,
            'description': "tokens to provide to the whisper decoder as initial prompt",
            'options': None,
            'default': None
    },
    'prompt_n_tokens': {
            'type': int,
            'description': "tokens to provide to the whisper decoder as initial prompt",
            'options': None,
            'default': 0
    },
    'carry_initial_prompt': {
            'type': bool,
            'description': "always prepend the initial prompt to each decode window",
            'options': None,
            'default': False
    },
    'language': {
            'type': str,
            'description': 'for auto-detection, set to None, "" or "auto"',
            'options': None,
            'default': "en"
    },
    'detect_language': {
            'type': bool,
            'description': 'enable automatic language detection during transcription',
            'options': None,
            'default': False
    },
    'suppress_blank': {
            'type': bool,
            'description': 'common decoding parameters',
            'options': None,
            'default': True
    },
    'suppress_non_speech_tokens': {
            'type': bool,
            'description': 'common decoding parameters',
            'options': None,
            'default': False
    },
    'suppress_nst': {
            'type': bool,
            'description': 'canonical whisper.cpp name for non-speech token suppression',
            'options': None,
            'default': False
    },
    'temperature': {
            'type': float,
            'description': 'initial decoding temperature',
            'options': None,
            'default': 0.0
    },
    'max_initial_ts': {
            'type': float,
            'description': 'max_initial_ts',
            'options': None,
            'default': 1.0
    },
    'length_penalty': {
            'type': float,
            'description': 'length_penalty',
            'options': None,
            'default': -1.0
    },
    'temperature_inc': {
            'type': float,
            'description': 'temperature_inc',
            'options': None,
            'default': 0.2
    },
    'entropy_thold': {
            'type': float,
            'description': 'similar to OpenAI\'s "compression_ratio_threshold"',
            'options': None,
            'default': 2.4
    },
    'logprob_thold': {
            'type': float,
            'description': 'logprob_thold',
            'options': None,
            'default': -1.0
    },
    'no_speech_thold': {  # not implemented
            'type': float,
            'description': 'no_speech_thold',
            'options': None,
            'default': 0.6
    },
    'grammar_penalty': {
            'type': float,
            'description': 'scales down logits of non-grammar tokens',
            'options': None,
            'default': 100.0
    },
    'greedy': {
            'type': dict,
            'description': 'greedy',
            'options': None,
            'default': {"best_of": 5}
    },
    'beam_search': {
            'type': dict,
            'description': 'beam_search',
            'options': None,
            'default': {"beam_size": 5, "patience": -1.0}
    },
    'extract_probability': {
            'type': bool,
            'description': 'calculate the geometric mean of token probabilities for each segment.',
            'options': None,
            'default': False
    },
    'vad': {
        'type': bool,
        'description': 'Enable VAD',
        'options': None,
        'default': False
    },
    'vad_model_path': {
        'type': str,
        'description': 'Path to VAD model',
        'options': None,
        'default': None
    }
}
