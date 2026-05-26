from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, TypedDict, TypeAlias, Union

import numpy as np
import numpy.typing as npt

AudioArray: TypeAlias = npt.NDArray[np.float32]
AudioInput: TypeAlias = Union[str, AudioArray]

class ContextParams(TypedDict, total=False):
    use_gpu: bool
    flash_attn: bool
    gpu_device: int
    dtw_token_timestamps: bool
    dtw_aheads_preset: int
    dtw_n_top: int
    dtw_mem_size: int


class GreedyParams(TypedDict):
    best_of: int


class BeamSearchParams(TypedDict):
    beam_size: int
    patience: float


class Segment:
    t0: int
    t1: int
    text: str
    probability: float

    def __init__(self, t0: int, t1: int, text: str, probability: float = np.nan)->None: ...
    def __str__(self)->str: ...
    def __repr__(self)->str: ...


class Model:
    model_path: str
    _new_segment_callback: Optional[Callable[[Segment], None]]

    def __init__(
        self,
        model: str = 'tiny',
        models_dir: Optional[str] = None,
        params_sampling_strategy: int = 0,
        redirect_whispercpp_logs_to: Union[bool, TextIO, str, None] = False,
        use_openvino: bool = False,
        openvino_model_path: Optional[str] = None,
        openvino_device: str = 'CPU',
        openvino_cache_dir: Optional[str] = None,
        context_params: Optional[ContextParams] = None,
        *,
        n_threads: Optional[int] = None,
        n_max_text_ctx: int = 16384,
        offset_ms: int = 0,
        duration_ms: int = 0,
        translate: bool = False,
        no_context: bool = True,
        no_timestamps: bool = False,
        single_segment: bool = False,
        print_special: bool = False,
        print_progress: bool = True,
        print_realtime: bool = False,
        print_timestamps: bool = True,
        token_timestamps: bool = False,
        thold_pt: float = 0.01,
        thold_ptsum: float = 0.01,
        max_len: int = 0,
        split_on_word: bool = False,
        max_tokens: int = 0,
        debug_mode: bool = False,
        audio_ctx: int = 0,
        tdrz_enable: bool = False,
        initial_prompt: Optional[str] = None,
        prompt_tokens: Optional[Tuple[Any, ...]] = None,
        prompt_n_tokens: int = 0,
        carry_initial_prompt: bool = False,
        language: str = '',
        detect_language: bool = False,
        suppress_blank: bool = True,
        suppress_non_speech_tokens: bool = False,
        suppress_nst: bool = False,
        suppress_regex: str = '',
        temperature: float = 0.0,
        max_initial_ts: float = 1.0,
        length_penalty: float = -1.0,
        temperature_inc: float = 0.2,
        entropy_thold: float = 2.4,
        logprob_thold: float = -1.0,
        no_speech_thold: float = 0.6,
        greedy: GreedyParams = {'best_of': 5},
        beam_search: BeamSearchParams = {'beam_size': -1, 'patience': -1.0},
        vad: bool = False,
        vad_model_path: Optional[str] = None,
        **params
    )->None: ...

    def transcribe(
        self,
        media: AudioInput,
        n_processors: Optional[int] = None,
        new_segment_callback: Optional[Callable[[Segment], None]] = None,
        abort_callback: Optional[Callable[[], bool]] = None,
        *,
        n_threads: Optional[int] = None,
        n_max_text_ctx: int = 16384,
        offset_ms: int = 0,
        duration_ms: int = 0,
        translate: bool = False,
        no_context: bool = True,
        no_timestamps: bool = False,
        single_segment: bool = False,
        print_special: bool = False,
        print_progress: bool = True,
        print_realtime: bool = False,
        print_timestamps: bool = True,
        token_timestamps: bool = False,
        thold_pt: float = 0.01,
        thold_ptsum: float = 0.01,
        max_len: int = 0,
        split_on_word: bool = False,
        max_tokens: int = 0,
        debug_mode: bool = False,
        audio_ctx: int = 0,
        tdrz_enable: bool = False,
        initial_prompt: Optional[str] = None,
        prompt_tokens: Optional[Tuple[Any, ...]] = None,
        prompt_n_tokens: int = 0,
        carry_initial_prompt: bool = False,
        language: str = '',
        detect_language: bool = False,
        suppress_blank: bool = True,
        suppress_non_speech_tokens: bool = False,
        suppress_nst: bool = False,
        suppress_regex: str = '',
        temperature: float = 0.0,
        max_initial_ts: float = 1.0,
        length_penalty: float = -1.0,
        temperature_inc: float = 0.2,
        entropy_thold: float = 2.4,
        logprob_thold: float = -1.0,
        no_speech_thold: float = 0.6,
        greedy: GreedyParams = {'best_of': 5},
        beam_search: BeamSearchParams = {'beam_size': -1, 'patience': -1.0},
        extract_probability: bool = False,
        vad: bool = False,
        vad_model_path: Optional[str] = None,
        **params
    ) -> List[Segment]: ...

    def get_params(self) -> Dict[str, Any]: ...
    @staticmethod
    def get_params_schema() -> Dict[str, Dict[str, Any]]: ...
    @staticmethod
    def lang_max_id() -> int: ...
    def print_timings(self) -> None: ...
    @staticmethod
    def system_info() -> Any: ...
    @staticmethod
    def available_languages() -> List[str]: ...
    @staticmethod
    def _load_audio(media_file_path: str) -> AudioArray: ...
    def auto_detect_language(
        self,
        media: AudioInput,
        offset_ms: Optional[int] = None,
        n_threads: Optional[int] = None,
    ) -> Tuple[Tuple[str, np.float32], Dict[str, np.float32]]: ...
    def __del__(self) -> None: ...

