# pywhispercpp
Python bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with a simple Pythonic API on top of it.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Wheels](https://github.com/absadiki/pywhispercpp/actions/workflows/wheels.yml/badge.svg?branch=main&event=push)](https://github.com/absadiki/pywhispercpp/actions/workflows/wheels.yml)
[![PyPi version](https://badgen.net/pypi/v/pywhispercpp)](https://pypi.org/project/pywhispercpp/)
[![Downloads](https://static.pepy.tech/badge/pywhispercpp)](https://pepy.tech/project/pywhispercpp)

# Table of contents
<!-- TOC -->
* [Installation](#installation)
    * [From source](#from-source)
    * [Pre-built wheels](#pre-built-wheels)
    * [NVIDIA GPU support](#nvidia-gpu-support)
    * [CoreML support](#coreml-support)
    * [Vulkan support](#vulkan-support)
* [Quick start](#quick-start)
* [Examples](#examples)
  * [CLI](#cli)
  * [Assistant](#assistant)
* [Advanced usage](#advanced-usage)
* [Discussions and contributions](#discussions-and-contributions)
* [License](#license)
<!-- TOC -->

# Installation

### From source
* For the best performance, you need to install the package from source:
```shell
pip install git+https://github.com/absadiki/pywhispercpp
```
### Pre-built wheels
* Otherwise, Basic Pre-built CPU wheels are available on PYPI

```shell
pip install pywhispercpp # or pywhispercpp[examples] to install the extra dependencies needed for the examples
```

[Optional] To transcribe files other than wav, you need to install ffmpeg:  
```shell
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

### NVIDIA GPU support
To Install the package with CUDA support, make sure you have [cuda](https://developer.nvidia.com/cuda-downloads) installed and use `GGML_CUDA=1`:

```shell
GGML_CUDA=1 pip install git+https://github.com/absadiki/pywhispercpp
```
### CoreML support

Install the package with `WHISPER_COREML=1`:

```shell
WHISPER_COREML=1 pip install git+https://github.com/absadiki/pywhispercpp
```

### Vulkan support

Install the package with `GGML_VULKAN=1`:

```shell
GGML_VULKAN=1 pip install git+https://github.com/absadiki/pywhispercpp
```

### OpenBLAS support

If OpenBLAS is installed, you can use `GGML_BLAS=1`. The other flags ensure you're installing fresh with the correct flags, and printing output for sanity checking.
```shell
GGML_BLAS=1 pip install git+https://github.com/absadiki/pywhispercpp --no-cache --force-reinstall -v
```

### OpenVINO support

Follow the the steps to download correct OpenVINO package (https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file#openvino-support).

Then init the OpenVINO environment and build.
```
source ~/l_openvino_toolkit_ubuntu22_2023.0.0.10926.b4452d56304_x86_64/setupvars.sh 
WHISPER_OPENVINO=1 pip install git+https://github.com/absadiki/pywhispercpp --no-cache --force-reinstall
```

Note that the toolkit for Ubuntu22 works on Ubuntu24


** __Feel free to update this list and submit a PR if you tested the package on other backends.__


# Quick start

```python
from pywhispercpp.model import Model

model = Model('base.en')
segments = model.transcribe('file.wav')
for segment in segments:
    print(segment.text)
```

You can also assign a custom `new_segment_callback`

```python
from pywhispercpp.model import Model

model = Model('base.en', print_realtime=False, print_progress=False)
segments = model.transcribe('file.mp3', new_segment_callback=print)
```


* The model will be downloaded automatically, or you can use the path to a local model.
* You can pass any `whisper.cpp` [parameter](https://absadiki.github.io/pywhispercpp/#pywhispercpp.constants.PARAMS_SCHEMA) as a keyword argument to the `Model` class or to the `transcribe` function.
* Check the [Model](https://absadiki.github.io/pywhispercpp/#pywhispercpp.model.Model) class documentation for more details.

# Examples

## CLI
Just a straightforward example Command Line Interface. 
You can use it as follows:

```shell
pwcpp file.wav -m base --output-srt --print_realtime true
```
Run ```pwcpp --help``` to get the help message

```shell
usage: pwcpp [-h] [-m MODEL] [--version] [--processors PROCESSORS] [-otxt] [-ovtt] [-osrt] [-ocsv] [--strategy STRATEGY]
             [--n_threads N_THREADS] [--n_max_text_ctx N_MAX_TEXT_CTX] [--offset_ms OFFSET_MS] [--duration_ms DURATION_MS]
             [--translate TRANSLATE] [--no_context NO_CONTEXT] [--single_segment SINGLE_SEGMENT] [--print_special PRINT_SPECIAL]
             [--print_progress PRINT_PROGRESS] [--print_realtime PRINT_REALTIME] [--print_timestamps PRINT_TIMESTAMPS]
             [--token_timestamps TOKEN_TIMESTAMPS] [--thold_pt THOLD_PT] [--thold_ptsum THOLD_PTSUM] [--max_len MAX_LEN]
             [--split_on_word SPLIT_ON_WORD] [--max_tokens MAX_TOKENS] [--audio_ctx AUDIO_CTX]
             [--prompt_tokens PROMPT_TOKENS] [--prompt_n_tokens PROMPT_N_TOKENS] [--language LANGUAGE] [--suppress_blank SUPPRESS_BLANK]
             [--suppress_non_speech_tokens SUPPRESS_NON_SPEECH_TOKENS] [--temperature TEMPERATURE] [--max_initial_ts MAX_INITIAL_TS]
             [--length_penalty LENGTH_PENALTY] [--temperature_inc TEMPERATURE_INC] [--entropy_thold ENTROPY_THOLD]
             [--logprob_thold LOGPROB_THOLD] [--no_speech_thold NO_SPEECH_THOLD] [--greedy GREEDY] [--beam_search BEAM_SEARCH]
             media_file [media_file ...]

positional arguments:
  media_file            The path of the media file or a list of filesseparated by space

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to the `ggml` model, or just the model name
  --version             show program's version number and exit
  --processors PROCESSORS
                        number of processors to use during computation
  -otxt, --output-txt   output result in a text file
  -ovtt, --output-vtt   output result in a vtt file
  -osrt, --output-srt   output result in a srt file
  -ocsv, --output-csv   output result in a CSV file
  --strategy STRATEGY   Available sampling strategiesGreefyDecoder -> 0BeamSearchDecoder -> 1
  --n_threads N_THREADS
                        Number of threads to allocate for the inferencedefault to min(4, available hardware_concurrency)
  --n_max_text_ctx N_MAX_TEXT_CTX
                        max tokens to use from past text as prompt for the decoder
  --offset_ms OFFSET_MS
                        start offset in ms
  --duration_ms DURATION_MS
                        audio duration to process in ms
  --translate TRANSLATE
                        whether to translate the audio to English
  --no_context NO_CONTEXT
                        do not use past transcription (if any) as initial prompt for the decoder
  --single_segment SINGLE_SEGMENT
                        force single segment output (useful for streaming)
  --print_special PRINT_SPECIAL
                        print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
  --print_progress PRINT_PROGRESS
                        print progress information
  --print_realtime PRINT_REALTIME
                        print results from within whisper.cpp (avoid it, use callback instead)
  --print_timestamps PRINT_TIMESTAMPS
                        print timestamps for each text segment when printing realtime
  --token_timestamps TOKEN_TIMESTAMPS
                        enable token-level timestamps
  --thold_pt THOLD_PT   timestamp token probability threshold (~0.01)
  --thold_ptsum THOLD_PTSUM
                        timestamp token sum probability threshold (~0.01)
  --max_len MAX_LEN     max segment length in characters
  --split_on_word SPLIT_ON_WORD
                        split on word rather than on token (when used with max_len)
  --max_tokens MAX_TOKENS
                        max tokens per segment (0 = no limit)
  --audio_ctx AUDIO_CTX
                        overwrite the audio context size (0 = use default)
  --prompt_tokens PROMPT_TOKENS
                        tokens to provide to the whisper decoder as initial prompt
  --prompt_n_tokens PROMPT_N_TOKENS
                        tokens to provide to the whisper decoder as initial prompt
  --language LANGUAGE   for auto-detection, set to None, "" or "auto"
  --suppress_blank SUPPRESS_BLANK
                        common decoding parameters
  --suppress_non_speech_tokens SUPPRESS_NON_SPEECH_TOKENS
                        common decoding parameters
  --temperature TEMPERATURE
                        initial decoding temperature
  --max_initial_ts MAX_INITIAL_TS
                        max_initial_ts
  --length_penalty LENGTH_PENALTY
                        length_penalty
  --temperature_inc TEMPERATURE_INC
                        temperature_inc
  --entropy_thold ENTROPY_THOLD
                        similar to OpenAI's "compression_ratio_threshold"
  --logprob_thold LOGPROB_THOLD
                        logprob_thold
  --no_speech_thold NO_SPEECH_THOLD
                        no_speech_thold
  --greedy GREEDY       greedy
  --beam_search BEAM_SEARCH
                        beam_search

```

## Assistant

This is a simple example showcasing the use of `pywhispercpp` to create an assistant like example. 
The idea is to use a Voice Activity Detector (VAD) to detect speech (in this example, we used webrtcvad), and when some speech is detected, we run the transcription. 
It is inspired from the [whisper.cpp/examples/command](https://github.com/ggerganov/whisper.cpp/tree/master/examples/command) example.

You can check the source code [here](https://github.com/absadiki/pywhispercpp/blob/main/pywhispercpp/examples/assistant.py) 
or you can use the class directly to create your own assistant:


```python
from pywhispercpp.examples.assistant import Assistant

my_assistant = Assistant(commands_callback=print, n_threads=8)
my_assistant.start()
```
Here, we set the `commands_callback` to a simple print function, so the commands will just get printed on the screen.

You can also run this example from the command line.
```shell
$ pwcpp-assistant --help

usage: pwcpp-assistant [-h] [-m MODEL] [-ind INPUT_DEVICE] [-st SILENCE_THRESHOLD] [-bd BLOCK_DURATION]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Whisper.cpp model, default to tiny.en
  -ind INPUT_DEVICE, --input_device INPUT_DEVICE
                        Id of The input device (aka microphone)
  -st SILENCE_THRESHOLD, --silence_threshold SILENCE_THRESHOLD
                        he duration of silence after which the inference will be running, default to 16
  -bd BLOCK_DURATION, --block_duration BLOCK_DURATION
                        minimum time audio updates in ms, default to 30
```
-------------

* Check the [examples folder](https://github.com/absadiki/pywhispercpp/tree/main/pywhispercpp/examples) for more examples.

# Advanced usage
* First check the [API documentation](https://absadiki.github.io/pywhispercpp/) for more advanced usage.
* If you are a more experienced user, you can access the exposed C-APIs directly from the binding module `_pywhispercpp`.

```python
import _pywhispercpp as pwcpp

ctx = pwcpp.whisper_init_from_file('path/to/ggml/model')
```

# Discussions and contributions
If you find any bug, please open an [issue](https://github.com/absadiki/pywhispercpp/issues).

If you have any feedback, or you want to share how you are using this project, feel free to use the [Discussions](https://github.com/absadiki/pywhispercpp/discussions) and open a new topic.

# License

This project is licensed under the same license as [whisper.cpp](https://github.com/ggerganov/whisper.cpp/blob/master/LICENSE) (MIT  [License](./LICENSE)).

