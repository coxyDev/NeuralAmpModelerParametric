# File: parametric_amp.py
# Complete parametric amp model — four cascaded stages
#
# Inherits from BaseNet to integrate with the NAM ecosystem:
# export, metadata, handshake, forward signature.
#
# Knob layout (8 knobs, each [0,1] internally, displayed 0-10 to user):
#   0: preamp_gain
#   1: bass
#   2: mid
#   3: treble
#   4: sag (power amp compression)
#   5: presence
#   6: depth
#   7: master_volume

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..base import BaseNet
from .stages import OutputStage, PowerAmpStage, PreampStage, ToneStackStage


class ParametricAmp(BaseNet):
    """
    Grey-box DDSP hybrid amp model with independently tunable stages.

    Architecture mirrors a real guitar amplifier:
        Input -> Preamp -> Tone Stack -> Power Amp -> Output -> Output

    Each stage uses differentiable DSP primitives (biquad filters,
    GRU nonlinearities, envelope followers) controlled by user-facing
    knobs via MLP knob controllers.

    At nominal settings (all knobs at 0.5), the model reproduces
    the captured amp tone. Moving knobs produces plausible variations
    constrained by the grey-box structure's physical priors.
    """

    NUM_KNOBS = 8
    KNOB_NAMES = [
        "preamp_gain",
        "bass",
        "mid",
        "treble",
        "sag",
        "presence",
        "depth",
        "master_volume",
    ]
    NOMINAL_VALUE = 0.5

    def __init__(
        self,
        preamp_config: Optional[dict] = None,
        tonestack_config: Optional[dict] = None,
        poweramp_config: Optional[dict] = None,
        output_config: Optional[dict] = None,
        sample_rate: Optional[float] = None,
    ):
        super().__init__(sample_rate=sample_rate)

        self._preamp_config = preamp_config or {}
        self._tonestack_config = tonestack_config or {}
        self._poweramp_config = poweramp_config or {}
        self._output_config = output_config or {}

        self._preamp = PreampStage(**self._preamp_config)
        self._tonestack = ToneStackStage(**self._tonestack_config)
        self._poweramp = PowerAmpStage(**self._poweramp_config)
        self._output = OutputStage(**self._output_config)

        # Stage bypass flags (not learned; set at inference time)
        self._bypass: Dict[str, bool] = {
            "preamp": False,
            "tonestack": False,
            "poweramp": False,
            "output": False,
        }

    # ---- Properties required by BaseNet ----

    @property
    def pad_start_default(self) -> bool:
        return False  # Recurrent model, no receptive field padding needed

    @property
    def receptive_field(self) -> int:
        return 1  # Sample-by-sample recurrent processing

    # ---- Forward pass ----

    def _forward(
        self,
        x: torch.Tensor,
        knobs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process audio through the four-stage amp chain.

        :param x: (B, L) input audio
        :param knobs: (B, 8) or (8,) knob values in [0, 1].
                      If None, uses nominal settings (0.5 for all).
        :return: (B, L) output audio
        """
        B, L = x.shape

        if knobs is None:
            knobs = torch.full(
                (self.NUM_KNOBS,), self.NOMINAL_VALUE, device=x.device
            )
        if knobs.ndim == 1:
            knobs = knobs.unsqueeze(0).expand(B, -1)

        y = x

        # Stage 1: Preamp (knob 0)
        if not self._bypass["preamp"]:
            y = self._preamp(y, knobs[:, 0:1])

        # Stage 2: Tone Stack (knobs 1-3)
        if not self._bypass["tonestack"]:
            y = self._tonestack(y, knobs[:, 1:4])

        # Stage 3: Power Amp (knobs 4-6)
        if not self._bypass["poweramp"]:
            y = self._poweramp(y, knobs[:, 4:7])

        # Stage 4: Output (knob 7)
        if not self._bypass["output"]:
            y = self._output(y, knobs[:, 7:8])

        return y

    # ---- Nominal settings for metadata ----

    def _at_nominal_settings(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate at default knob positions (all 0.5)."""
        knobs = torch.full(
            (self.NUM_KNOBS,), self.NOMINAL_VALUE, device=x.device
        )
        return self(x, knobs=knobs)

    # ---- Bypass control ----

    def set_bypass(self, stage: str, bypass: bool):
        """Enable/disable bypass for a specific stage at inference time."""
        if stage not in self._bypass:
            raise ValueError(
                f"Unknown stage '{stage}'. Valid stages: {list(self._bypass.keys())}"
            )
        self._bypass[stage] = bypass

    def get_bypass(self, stage: str) -> bool:
        return self._bypass[stage]

    # ---- Export interface ----

    def _export_config(self) -> dict:
        return {
            "preamp": self._preamp_config,
            "tonestack": self._tonestack_config,
            "poweramp": self._poweramp_config,
            "output": self._output_config,
            "num_knobs": self.NUM_KNOBS,
            "knob_names": list(self.KNOB_NAMES),
        }

    def _export_weights(self) -> np.ndarray:
        """Flatten all stage weights into a single 1D array."""
        tensors = []
        for stage in [self._preamp, self._tonestack, self._poweramp, self._output]:
            for param in stage.parameters():
                tensors.append(param.data.cpu().flatten())
        return torch.cat(tensors).numpy()

    def _get_export_architecture(self) -> str:
        return "ParametricAmp"

    def _export_input_output_args(self) -> Tuple:
        """Provide nominal knobs for the export snapshot."""
        return (torch.full((self.NUM_KNOBS,), self.NOMINAL_VALUE),)

    def _export_input_output(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create example input/output with nominal knobs."""
        args = self._export_input_output_args()
        rate = self.sample_rate
        if rate is None:
            raise RuntimeError(
                "Cannot export model's input and output without a sample rate."
            )
        rate = int(rate)
        x = torch.cat(
            [
                torch.zeros((rate,)),
                0.5
                * torch.sin(
                    2.0
                    * 3.14159265
                    * 220.0
                    * torch.linspace(0.0, 1.0, rate + 1)[:-1]
                ),
                torch.zeros((rate,)),
            ]
        )
        knobs = args[0]
        with torch.no_grad():
            y = self(x, knobs=knobs, pad_start=True)
        return x.numpy(), y.numpy()

    def _get_non_user_metadata(self) -> dict:
        d = super()._get_non_user_metadata()
        d["parametric"] = {
            "num_knobs": self.NUM_KNOBS,
            "knob_names": list(self.KNOB_NAMES),
            "knob_defaults": [self.NOMINAL_VALUE] * self.NUM_KNOBS,
            "knob_ranges": [[0.0, 1.0]] * self.NUM_KNOBS,
            "stages": list(self._bypass.keys()),
        }
        return d

    # ---- Weight import ----

    def import_weights(self, weights: Sequence[float]):
        """Load weights from a flat array (inverse of _export_weights)."""
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        offset = 0
        for stage in [self._preamp, self._tonestack, self._poweramp, self._output]:
            for param in stage.parameters():
                n = param.numel()
                param.data = weights_tensor[offset : offset + n].reshape(
                    param.shape
                )
                offset += n

    # ---- Config parsing for InitializableFromConfig ----

    @classmethod
    def parse_config(cls, config):
        return config

    @classmethod
    def init_from_config(cls, config):
        return cls(**config)
