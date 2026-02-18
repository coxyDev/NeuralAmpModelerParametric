# File: stages.py
# Individual amp stages for the parametric amp model
#
# Four-stage cascade mirroring real amp topology:
#   1. Preamp: Wiener-Hammerstein (filter -> nonlinearity -> filter)
#   2. Tone Stack: Parametric EQ (low shelf, peak, high shelf)
#   3. Power Amp: WH + envelope-driven sag + negative feedback
#   4. Output: Transformer hysteresis + filtering + master volume

import math
from typing import Optional

import torch
import torch.nn as nn

from .dsp_primitives import (
    BiquadCascade,
    DifferentiableEnvelopeFollower,
    GRUNonlinearity,
    KnobController,
    ParametricBiquad,
    StableBiquadController,
)


class PreampStage(nn.Module):
    """
    Wiener-Hammerstein preamp model.

    Signal flow:
        input -> [biquad_pre] -> [gain] -> [GRU nonlinearity] -> [biquad_post] -> output

    The biquad filters capture frequency-dependent behavior before and after
    the tube's nonlinearity. The GRU captures dynamic, history-dependent
    distortion character (bias drift, asymmetric clipping).

    Controlled by: preamp_gain knob (1 value in [0, 1])
    The knob controller maps gain to:
    - gain_multiplier: applied before the GRU (scales input into nonlinearity)
    - bias_offset: added to input of GRU (shifts operating point)
    """

    def __init__(
        self,
        num_biquad_sections_pre: int = 2,
        num_biquad_sections_post: int = 2,
        gru_hidden_size: int = 1,
    ):
        super().__init__()
        self.num_biquad_sections_pre = num_biquad_sections_pre
        self.num_biquad_sections_post = num_biquad_sections_post
        self.gru_hidden_size = gru_hidden_size

        self._pre_filter = BiquadCascade(num_biquad_sections_pre)
        self._post_filter = BiquadCascade(num_biquad_sections_post)
        self._nonlinearity = GRUNonlinearity(hidden_size=gru_hidden_size)

        # Knob controller: 1 knob -> 2 outputs (gain_multiplier, bias_offset)
        self._knob_ctrl = KnobController(knob_dim=1, output_dim=2, hidden_dim=16)

    def forward(self, x: torch.Tensor, knobs: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, L) audio
        :param knobs: (B, 1) preamp gain knob [0, 1]
        :return: (B, L) preamp output
        """
        # Get gain parameters from knob
        params = self._knob_ctrl(knobs)  # (B, 2)
        # Gain multiplier: exp mapping for wide range, centered around 1.0
        gain_mult = torch.exp(params[:, 0:1] * 4.0 - 2.0)  # range ~[0.13, 7.4]
        bias = params[:, 1:2] * 0.1  # small bias offset

        # Pre-filter
        y = self._pre_filter(x)

        # Apply gain and bias
        y = y * gain_mult + bias

        # Nonlinear stage
        y = self._nonlinearity(y)

        # Post-filter
        y = self._post_filter(y)

        return y


class ToneStackStage(nn.Module):
    """
    Three-band parametric EQ controlled by bass/mid/treble knobs.

    Signal flow:
        input -> [low_shelf] -> [parametric_peak] -> [high_shelf] -> output

    Each biquad's coefficients are predicted by a StableBiquadController MLP
    that takes the 3 tone knob values and produces stable filter coefficients.

    This mirrors real Fender/Marshall/Vox tone stacks where passive components
    create frequency-dependent attenuation controlled by pots.

    Controlled by: bass, mid, treble knobs (3 values in [0, 1])
    """

    def __init__(self, sample_rate: float = 48000.0):
        super().__init__()
        self.sample_rate = sample_rate

        # 3 knobs -> 3 biquad sections (low shelf, peak, high shelf)
        self._biquad_ctrl = StableBiquadController(
            knob_dim=3, num_sections=3, hidden_dim=16
        )

    def forward(self, x: torch.Tensor, knobs: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, L) audio
        :param knobs: (B, 3) [bass, mid, treble] in [0, 1]
        :return: (B, L) EQ'd audio
        """
        # Get stable biquad coefficients from knobs
        coeffs = self._biquad_ctrl(knobs)  # (B, 3, 5)

        # Apply each biquad section in series
        y = x
        for i in range(3):
            y = ParametricBiquad.forward(y, coeffs[:, i, :])

        return y


class PowerAmpStage(nn.Module):
    """
    Power amp model with sag/compression and negative feedback.

    Signal flow:
        input -> [envelope_follower] -> sag_gain modulation
              -> [biquad_pre] -> [GRU nonlinearity] -> [biquad_post]
              -> [negative_feedback_mix] -> output

    The envelope follower detects signal level. At high levels, sag_gain
    reduces (simulating power supply voltage drop), compressing the signal
    and increasing distortion.

    Presence controls high-frequency content in the negative feedback path.
    Depth controls low-frequency content in the negative feedback path.

    Controlled by: sag, presence, depth knobs (3 values in [0, 1])
    """

    def __init__(
        self,
        gru_hidden_size: int = 1,
        num_biquad_sections: int = 2,
    ):
        super().__init__()
        self.gru_hidden_size = gru_hidden_size
        self.num_biquad_sections = num_biquad_sections

        # Core WH chain
        self._pre_filter = BiquadCascade(num_biquad_sections)
        self._nonlinearity = GRUNonlinearity(hidden_size=gru_hidden_size)
        self._post_filter = BiquadCascade(num_biquad_sections)

        # Envelope follower for sag
        self._envelope = DifferentiableEnvelopeFollower()

        # Knob controllers:
        # sag knob -> sag_depth (how much the envelope affects gain)
        self._sag_ctrl = KnobController(knob_dim=1, output_dim=1, hidden_dim=8)
        # presence + depth -> feedback filter coefficients (1 biquad for each)
        self._feedback_ctrl = StableBiquadController(
            knob_dim=2, num_sections=1, hidden_dim=16
        )

        # Learnable feedback mix amount
        self._feedback_amount = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, knobs: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, L) audio
        :param knobs: (B, 3) [sag, presence, depth] in [0, 1]
        :return: (B, L) power amp output
        """
        sag_knob = knobs[:, 0:1]
        feedback_knobs = knobs[:, 1:3]  # presence, depth

        # Envelope detection
        env = self._envelope(x)  # (B, L) envelope

        # Sag: reduce gain when envelope is high
        sag_depth = torch.sigmoid(self._sag_ctrl(sag_knob))  # (B, 1) in [0, 1]
        # sag_gain = 1.0 when env=0, decreases toward (1-sag_depth) at high env
        sag_gain = 1.0 - sag_depth * env  # (B, L)

        y = x * sag_gain

        # WH core
        y = self._pre_filter(y)
        y = self._nonlinearity(y)
        y = self._post_filter(y)

        # Negative feedback: presence/depth shape the feedback frequency content
        fb_coeffs = self._feedback_ctrl(feedback_knobs)  # (B, 1, 5)
        fb_signal = ParametricBiquad.forward(y, fb_coeffs[:, 0, :])
        fb_mix = torch.sigmoid(self._feedback_amount)
        y = y - fb_mix * fb_signal

        return y


class OutputStage(nn.Module):
    """
    Output transformer and master volume.

    Signal flow:
        input -> [GRU nonlinearity] -> [biquad_filter] -> [master_volume] -> output

    The GRU captures transformer hysteresis (output transformer's nonlinear
    behavior, which is subtly different from tube distortion — it includes
    core saturation and hysteresis effects that are history-dependent).

    The biquad captures the transformer's frequency response rolloff.

    Controlled by: master_volume knob (1 value in [0, 1])
    """

    def __init__(self, gru_hidden_size: int = 1):
        super().__init__()
        self.gru_hidden_size = gru_hidden_size

        self._nonlinearity = GRUNonlinearity(hidden_size=gru_hidden_size)
        self._filter = BiquadCascade(num_sections=1)

        # Master volume: knob -> gain multiplier
        self._volume_ctrl = KnobController(knob_dim=1, output_dim=1, hidden_dim=8)

    def forward(self, x: torch.Tensor, knobs: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, L) audio
        :param knobs: (B, 1) [master_volume] in [0, 1]
        :return: (B, L) output
        """
        y = self._nonlinearity(x)
        y = self._filter(y)

        # Master volume with smooth exponential mapping
        vol_param = self._volume_ctrl(knobs)  # (B, 1)
        volume = torch.exp(vol_param * 4.0 - 4.0)  # range ~[0.018, 1.0]
        y = y * volume

        return y
