#!/usr/bin/env python

"""
Helper functions
"""

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import TextIO

import requests
from tqdm import tqdm

from pywhispercpp.constants import (
    AVAILABLE_MODELS,
    MODELS_BASE_URL,
    MODELS_DIR,
    MODELS_PREFIX_URL,
)

logger = logging.getLogger(__name__)


def _get_model_url(model_name: str) -> str:
    """
    Returns the url of the `ggml` model
    :param model_name: name of the model
    :return: URL of the model
    """
    return f"{MODELS_BASE_URL}/{MODELS_PREFIX_URL}-{model_name}.bin"


def download_model(model_name: str, download_dir=None, chunk_size=1024) -> str:
    """
    Helper function to download the `ggml` models
    :param model_name: name of the model, one of ::: constants.AVAILABLE_MODELS
    :param download_dir: Where to store the models
    :param chunk_size: size of the download chunk

    :return: Absolute path of the downloaded model
    """
    if model_name not in AVAILABLE_MODELS:
        logger.error(f"Invalid model name `{model_name}`, available models are: {AVAILABLE_MODELS}")
        return
    if download_dir is None:
        download_dir = MODELS_DIR
        logger.info(f"No download directory was provided, models will be downloaded to {download_dir}")

    os.makedirs(download_dir, exist_ok=True)

    url = _get_model_url(model_name=model_name)
    file_path = Path(download_dir) / os.path.basename(url)
    # check if the file is already there
    if file_path.exists():
        logger.info(f"Model {model_name} already exists in {download_dir}")
    else:
        # download it from huggingface
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))

        progress_bar = tqdm(desc=f"Downloading Model {model_name} ...",
                            total=total,
                            unit='iB',
                            unit_scale=True,
                            unit_divisor=1024)

        try:
            with open(file_path, 'wb') as file, progress_bar:
                for data in resp.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    progress_bar.update(size)
            logger.info(f"Model downloaded to {file_path.absolute()}")
        except Exception as e:
            # error download, just remove the file
            os.remove(file_path)
            raise e
    return str(file_path.absolute())


def resolve_model_path(model_name: str, models_dir=None) -> str:
    """
    Resolve a model name to a local model file.

    Resolution order:
    1. If `model_name` is an existing file path, return it.
    2. Look for `model_name` and `model_name.bin` in `models_dir`.
    3. If no local file is found, fall back to downloading a built-in model.

    :param model_name: A built-in model name, a custom model name, or a direct path to a model file.
    :param models_dir: Directory to search for local models before downloading. Defaults to `MODELS_DIR`.
    :return: Absolute path to the resolved model file.
    """
    if Path(model_name).is_file():
        return str(Path(model_name).resolve())

    search_dir = Path(models_dir) if models_dir is not None else MODELS_DIR

    candidates = [
        search_dir / model_name,
        search_dir / f"{model_name}.bin",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    return download_model(model_name, search_dir)


def to_timestamp(t: int, separator=',') -> str:
    """
    376 -> 00:00:03,760
    1344 -> 00:00:13,440

    Implementation from `whisper.cpp/examples/main`

    :param t: input time from whisper timestamps
    :param separator: seprator between seconds and milliseconds
    :return: time representation in hh: mm: ss[separator]ms
    """
    # logic exactly from whisper.cpp

    msec = t * 10
    hr = msec // (1000 * 60 * 60)
    msec = msec - hr * (1000 * 60 * 60)
    min = msec // (1000 * 60)
    msec = msec - min * (1000 * 60)
    sec = msec // 1000
    msec = msec - sec * 1000
    return f"{int(hr):02,.0f}:{int(min):02,.0f}:{int(sec):02,.0f}{separator}{int(msec):03,.0f}"


def output_txt(segments: list, output_file_path: str) -> str:
    """
    Creates a raw text from a list of segments

    Implementation from `whisper.cpp/examples/main`

    :param segments: list of segments
    :return: path of the file
    """
    if not output_file_path.endswith('.txt'):
        output_file_path = output_file_path + '.txt'

    absolute_path = Path(output_file_path).absolute()

    with open(str(absolute_path), 'w') as file:
        for seg in segments:
            file.write(seg.text)
            file.write('\n')
    return absolute_path


def output_vtt(segments: list, output_file_path: str) -> str:
    """
    Creates a vtt file from a list of segments

    Implementation from `whisper.cpp/examples/main`

    :param segments: list of segments
    :return: path of the file

    :return: Absolute path of the file
    """
    if not output_file_path.endswith('.vtt'):
        output_file_path = output_file_path + '.vtt'

    absolute_path = Path(output_file_path).absolute()

    with open(absolute_path, 'w') as file:
        file.write("WEBVTT\n\n")
        for seg in segments:
            file.write(f"{to_timestamp(seg.t0, separator='.')} --> {to_timestamp(seg.t1, separator='.')}\n")
            file.write(f"{seg.text}\n\n")
    return absolute_path


def output_srt(segments: list, output_file_path: str) -> str:
    """
    Creates a srt file from a list of segments

    :param segments: list of segments
    :return: path of the file

    :return: Absolute path of the file
    """
    if not output_file_path.endswith('.srt'):
        output_file_path = output_file_path + '.srt'

    absolute_path = Path(output_file_path).absolute()

    with open(absolute_path, 'w') as file:
        for i in range(len(segments)):
            seg = segments[i]
            file.write(f"{i+1}\n")
            file.write(f"{to_timestamp(seg.t0, separator=',')} --> {to_timestamp(seg.t1, separator=',')}\n")
            file.write(f"{seg.text}\n\n")
    return absolute_path


def output_csv(segments: list, output_file_path: str) -> str:
    """
    Creates a srt file from a list of segments

    :param segments: list of segments
    :return: path of the file

    :return: Absolute path of the file
    """
    if not output_file_path.endswith('.csv'):
        output_file_path = output_file_path + '.csv'

    absolute_path = Path(output_file_path).absolute()

    with open(absolute_path, 'w') as file:
        for seg in segments:
            file.write(f"{10 * seg.t0}, {10 * seg.t1}, \"{seg.text}\"\n")
    return absolute_path


@contextlib.contextmanager
def redirect_stderr(to: bool | TextIO | str | None = False) -> None:
    """
    Redirect stderr to the specified target.

    :param to:
        - None to suppress output (redirect to devnull),
        - sys.stdout to redirect to stdout,
        - A file path (str) to redirect to a file,
        - False to do nothing (no redirection).
    """

    if to is False:
        # do nothing
        yield
        return

    def _resolve_target(target):
        opened_stream = None
        if target is None:
            opened_stream = open(os.devnull, "w")
            return opened_stream, True
        if isinstance(target, str):
            opened_stream = open(target, "w")
            return opened_stream, True
        if hasattr(target, "write"):
            return target, False
        raise ValueError(
            "Invalid `to` parameter; expected None, a filepath string, or a file-like object."
        )

    sys.stderr.flush()
    try:
        original_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        # Jupyter or non-standard stderr implementations
        original_fd = None

    stream, should_close = _resolve_target(to)

    if original_fd is not None and hasattr(stream, "fileno"):
        saved_fd = os.dup(original_fd)
        try:
            os.dup2(stream.fileno(), original_fd)
            yield
        finally:
            os.dup2(saved_fd, original_fd)
            os.close(saved_fd)
            if should_close:
                stream.close()
        return

    # Fallback: Python-level redirect
    try:
        with contextlib.redirect_stderr(stream):
            yield
    finally:
        if should_close:
            stream.close()
