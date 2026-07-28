from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class Segment:
    """
    A small class representing a transcription segment
    """

    def __init__(self, t0: int, t1: int, text: str, probability: float = np.nan):
        """
        :param t0: start time
        :param t1: end time
        :param text: text
        :param probability: Confidence score for the segment, computed as the geometric mean of
            the token probabilities for the segment (NaN if not calculated).
            This makes it interpretable as a probability in [0, 1].
        """
        self.t0 = t0
        self.t1 = t1
        self.text = text
        self.probability = probability

    def __str__(self):
        return f"t0={self.t0}, t1={self.t1}, text={self.text}, probability={self.probability}"

    def __repr__(self):
        return str(self)


class BaseModel(ABC):
    """
    Abstract base class for all transcription models (whisper, parakeet, etc.).
    Defines the public contract that every model must implement.
    """

    @abstractmethod
    def transcribe(self, media: Union[str, np.ndarray], **params) -> List[Segment]:
        """
        Transcribe audio media and return a list of segments.
        :param media: file path or numpy array of audio data
        :return: list of Segment objects
        """
        ...
