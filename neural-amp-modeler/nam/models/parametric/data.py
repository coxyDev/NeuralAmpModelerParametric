# File: data.py
# Dataset for parametric amp training
#
# Unlike the standard Dataset which returns (x, y) tuples,
# this returns (x, knobs, y) tuples where knobs is a fixed
# vector representing the amp settings at capture time.

from typing import Optional, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset as _TorchDataset

from ..._core import InitializableFromConfig


class ParametricDataset(_TorchDataset, InitializableFromConfig):
    """
    Dataset for parametric amp training.

    Returns (x_segment, knobs, y_segment) tuples.

    During initial training (profiling), knobs are fixed at nominal
    (0.5 for all) because the amp was profiled at one setting.

    The knob vector is included in every batch item so the model
    always receives it, enabling future knob augmentation.
    """

    NUM_KNOBS = 8

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        nx: int,
        ny: Optional[int] = None,
        knobs: Optional[torch.Tensor] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        delay: Optional[Union[int, float]] = None,
        y_scale: float = 1.0,
        input_gain: float = 0.0,
        sample_rate: Optional[float] = None,
    ):
        """
        :param x: (N,) input signal
        :param y: (N,) output signal
        :param nx: Receptive field of the model (1 for ParametricAmp)
        :param ny: Output segment length per datum
        :param knobs: (NUM_KNOBS,) knob settings. Defaults to 0.5 for all.
        :param start: Clip signals from this sample
        :param stop: Clip signals to this sample
        :param delay: Alignment correction in samples
        :param y_scale: Output gain multiplier
        :param input_gain: Input gain in dB
        :param sample_rate: Sample rate of the data
        """
        if start is not None or stop is not None:
            s = start or 0
            e = stop or len(x)
            x = x[s:e]
            y = y[s:e]

        if delay is not None and delay != 0:
            delay = int(delay)
            if delay > 0:
                x = x[delay:]
                y = y[:-delay]
            else:
                x = x[:delay]
                y = y[-delay:]

        if input_gain != 0.0:
            x = x * (10.0 ** (input_gain / 20.0))
        y = y * y_scale

        self._x = x
        self._y = y
        self._nx = nx
        self._ny = ny if ny is not None else len(x) - nx + 1
        self._sample_rate = sample_rate

        if knobs is not None:
            self._knobs = knobs
        else:
            self._knobs = torch.full((self.NUM_KNOBS,), 0.5)

    def __len__(self) -> int:
        n = len(self._x)
        single_pairs = n - self._nx + 1
        return single_pairs // self._ny

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return: (x_segment, knobs, y_segment)
            x_segment: (NX+NY-1,)
            knobs: (NUM_KNOBS,)
            y_segment: (NY,)
        """
        if idx >= len(self):
            raise IndexError(
                f"Attempted to access datum {idx}, but len is {len(self)}"
            )
        i = idx * self._ny
        x_seg = self._x[i : i + self._nx + self._ny - 1]
        y_seg = self._y[i : i + self._ny]
        return x_seg, self._knobs.clone(), y_seg

    @property
    def sample_rate(self) -> Optional[float]:
        return self._sample_rate

    @classmethod
    def parse_config(cls, config):
        return config

    @classmethod
    def init_from_config(cls, config):
        return cls(**config)
