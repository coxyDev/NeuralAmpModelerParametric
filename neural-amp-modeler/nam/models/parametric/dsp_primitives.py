# File: dsp_primitives.py
# Differentiable DSP primitives for parametric amp modelling
#
# These are the atomic building blocks of the grey-box amp model:
# - Biquad IIR filters (learnable and externally-controlled)
# - GRU nonlinearity (dynamic tube-like distortion)
# - Envelope follower (power amp sag detection)
# - Knob controller (MLP mapping user knobs to DSP parameters)

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class DifferentiableBiquad(nn.Module):
    """
    Second-order IIR filter with learnable coefficients.

    Transfer function:
        H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)

    Uses Direct Form II Transposed for numerical stability.
    Processes sample-by-sample for gradient flow through the recursion.

    Stability is enforced by parameterizing poles as (radius, angle):
        radius in (0, max_radius) via sigmoid
        angle in (0, pi) via sigmoid * pi
        a1 = -2 * radius * cos(angle)
        a2 = radius^2

    This guarantees both poles remain inside the unit circle during training.
    """

    MAX_RADIUS = 0.999

    def __init__(self, num_filters: int = 1):
        super().__init__()
        self.num_filters = num_filters
        # Pole parameterization (unconstrained, mapped through sigmoid)
        self._log_radius = nn.Parameter(torch.zeros(num_filters))
        self._raw_angle = nn.Parameter(torch.zeros(num_filters))
        # Feed-forward coefficients (unconstrained)
        self.b0 = nn.Parameter(torch.ones(num_filters))
        self.b1 = nn.Parameter(torch.zeros(num_filters))
        self.b2 = nn.Parameter(torch.zeros(num_filters))

    def get_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute stable (b, a) coefficients from the parameterization.

        :return: (b_coeffs (num_filters, 3), a_coeffs (num_filters, 2))
                 where a_coeffs are [a1, a2] (a0=1 implied)
        """
        radius = torch.sigmoid(self._log_radius) * self.MAX_RADIUS
        angle = torch.sigmoid(self._raw_angle) * math.pi
        a1 = -2.0 * radius * torch.cos(angle)
        a2 = radius * radius
        b = torch.stack([self.b0, self.b1, self.b2], dim=-1)
        a = torch.stack([a1, a2], dim=-1)
        return b, a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the biquad filter(s) to the input signal.

        :param x: (B, L) mono audio
        :return: (B, L) filtered audio

        If num_filters > 1, filters are applied in series (cascade).
        """
        b, a = self.get_coefficients()

        for f in range(self.num_filters):
            x = self._apply_single_biquad(
                x, b[f, 0], b[f, 1], b[f, 2], a[f, 0], a[f, 1]
            )
        return x

    @staticmethod
    def _apply_single_biquad(
        x: torch.Tensor,
        b0: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Direct Form II Transposed biquad, sample-by-sample.

        State equations:
            y[n] = b0*x[n] + s1[n-1]
            s1[n] = b1*x[n] - a1*y[n] + s2[n-1]
            s2[n] = b2*x[n] - a2*y[n]
        """
        B, L = x.shape
        device = x.device

        s1 = torch.zeros(B, device=device)
        s2 = torch.zeros(B, device=device)
        outputs = []

        for n in range(L):
            xn = x[:, n]
            yn = b0 * xn + s1
            s1 = b1 * xn - a1 * yn + s2
            s2 = b2 * xn - a2 * yn
            outputs.append(yn)

        return torch.stack(outputs, dim=1)


class BiquadCascade(nn.Module):
    """
    Chain of N independent biquad filter sections in series.
    Each section has its own learnable coefficients.
    """

    def __init__(self, num_sections: int):
        super().__init__()
        self.num_sections = num_sections
        self.sections = nn.ModuleList(
            [DifferentiableBiquad(num_filters=1) for _ in range(num_sections)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, L) audio
        :return: (B, L) filtered audio
        """
        for section in self.sections:
            x = section(x)
        return x


class ParametricBiquad(nn.Module):
    """
    Biquad filter whose coefficients are provided externally
    (e.g. predicted by a KnobController MLP).

    Does not have its own learnable coefficients — the coefficients
    are an input to the forward pass.

    Stability must be enforced by the caller (e.g. via radius/angle
    parameterization in the KnobController output).
    """

    @staticmethod
    def forward(
        x: torch.Tensor, coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: (B, L) audio
        :param coeffs: (B, 5) or (5,) biquad coefficients [b0, b1, b2, a1, a2]
        :return: (B, L) filtered audio
        """
        if coeffs.ndim == 1:
            coeffs = coeffs.unsqueeze(0).expand(x.shape[0], -1)

        b0 = coeffs[:, 0]
        b1 = coeffs[:, 1]
        b2 = coeffs[:, 2]
        a1 = coeffs[:, 3]
        a2 = coeffs[:, 4]

        B, L = x.shape
        device = x.device
        s1 = torch.zeros(B, device=device)
        s2 = torch.zeros(B, device=device)
        outputs = []

        for n in range(L):
            xn = x[:, n]
            yn = b0 * xn + s1
            s1 = b1 * xn - a1 * yn + s2
            s2 = b2 * xn - a2 * yn
            outputs.append(yn)

        return torch.stack(outputs, dim=1)


class GRUNonlinearity(nn.Module):
    """
    Tiny GRU cell used as a dynamic, history-dependent nonlinearity.

    With hidden_size=1, this reduces to a single recurrent scalar state
    that captures the memory-dependent distortion behavior of vacuum tubes
    (bias-point drift, thermal inertia, asymmetric clipping that depends
    on recent signal history).

    Much more expressive than static tanh/clipping for the same compute cost.
    """

    def __init__(self, hidden_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self._gru = nn.GRUCell(input_size=1, hidden_size=hidden_size)
        self._head = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        :param x: (B, L) audio
        :param hidden: (B, hidden_size) initial hidden state, or None for zeros
        :return: (B, L) nonlinearly processed audio
        """
        B, L = x.shape
        device = x.device

        if hidden is None:
            hidden = torch.zeros(B, self.hidden_size, device=device)

        outputs = []
        for n in range(L):
            xn = x[:, n : n + 1]  # (B, 1)
            hidden = self._gru(xn, hidden)
            yn = self._head(hidden)  # (B, 1)
            outputs.append(yn[:, 0])

        return torch.stack(outputs, dim=1)


class DifferentiableEnvelopeFollower(nn.Module):
    """
    Learnable RMS/peak envelope detector for power amp sag simulation.

    Uses learnable attack and release time constants. The envelope tracks
    the absolute signal level:
        - When signal rises: env follows with attack time constant
        - When signal falls: env follows with release time constant

    Output is a smooth envelope signal in [0, 1+] range.
    """

    def __init__(self):
        super().__init__()
        # Unconstrained params, mapped through sigmoid to get coefficients
        # Default ~10ms attack, ~100ms release at 48kHz
        self._raw_attack = nn.Parameter(torch.tensor(0.0))
        self._raw_release = nn.Parameter(torch.tensor(-2.0))

    def _get_coefficients(self, sample_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert raw params to per-sample smoothing coefficients in (0, 1)."""
        # Map to time constants in ms, then to per-sample coefficients
        # attack: fast (0.1ms to 50ms), release: slow (10ms to 500ms)
        attack_ms = 0.1 + torch.sigmoid(self._raw_attack) * 49.9
        release_ms = 10.0 + torch.sigmoid(self._raw_release) * 490.0

        # Coefficient = exp(-1 / (time_constant_samples))
        attack_samples = attack_ms * sample_rate / 1000.0
        release_samples = release_ms * sample_rate / 1000.0

        attack_coeff = torch.exp(-1.0 / attack_samples)
        release_coeff = torch.exp(-1.0 / release_samples)

        return attack_coeff, release_coeff

    def forward(
        self, x: torch.Tensor, sample_rate: float = 48000.0
    ) -> torch.Tensor:
        """
        :param x: (B, L) audio input
        :param sample_rate: sample rate in Hz
        :return: (B, L) envelope signal (non-negative)
        """
        attack_coeff, release_coeff = self._get_coefficients(sample_rate)

        B, L = x.shape
        device = x.device
        env = torch.zeros(B, device=device)
        abs_x = torch.abs(x)
        outputs = []

        for n in range(L):
            level = abs_x[:, n]
            # Select attack or release coefficient based on signal vs envelope
            coeff = torch.where(level > env, attack_coeff, release_coeff)
            env = coeff * env + (1.0 - coeff) * level
            outputs.append(env)

        return torch.stack(outputs, dim=1)


class KnobController(nn.Module):
    """
    MLP that maps user-facing knob values [0, 1] to internal DSP parameters.

    Small network: knob_dim -> hidden -> hidden -> output_dim

    The output_dim depends on what DSP parameters are being controlled:
    - For biquad coefficients: 5 per filter section (b0, b1, b2, a1, a2)
      but raw outputs get stability-enforced before use
    - For gain values: 1
    - For bias offsets: 1

    Uses Tanh activations to keep internal representations bounded.
    """

    def __init__(self, knob_dim: int, output_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.knob_dim = knob_dim
        self.output_dim = output_dim
        self._net = nn.Sequential(
            nn.Linear(knob_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, knobs: torch.Tensor) -> torch.Tensor:
        """
        :param knobs: (B, knob_dim) or (knob_dim,) knob values in [0, 1]
        :return: (B, output_dim) or (output_dim,) DSP parameters
        """
        squeeze = knobs.ndim == 1
        if squeeze:
            knobs = knobs.unsqueeze(0)
        out = self._net(knobs)
        if squeeze:
            out = out.squeeze(0)
        return out


class StableBiquadController(nn.Module):
    """
    Wraps a KnobController to produce stable biquad coefficients.

    Takes raw knob values, passes through the MLP, then enforces stability
    on the predicted filter coefficients using the radius/angle parameterization.

    Produces coefficients for N biquad sections.
    """

    MAX_RADIUS = 0.999

    def __init__(self, knob_dim: int, num_sections: int, hidden_dim: int = 16):
        super().__init__()
        self.num_sections = num_sections
        # Each section needs 5 raw outputs: b0, b1, b2, raw_radius, raw_angle
        self._controller = KnobController(
            knob_dim, num_sections * 5, hidden_dim
        )

    def forward(self, knobs: torch.Tensor) -> torch.Tensor:
        """
        :param knobs: (B, knob_dim) or (knob_dim,)
        :return: (B, num_sections, 5) stable biquad coefficients [b0,b1,b2,a1,a2]
                 or (num_sections, 5) if input was 1D
        """
        squeeze = knobs.ndim == 1
        if squeeze:
            knobs = knobs.unsqueeze(0)

        raw = self._controller(knobs)  # (B, num_sections * 5)
        B = raw.shape[0]
        raw = raw.view(B, self.num_sections, 5)

        b0 = raw[:, :, 0]
        b1 = raw[:, :, 1]
        b2 = raw[:, :, 2]

        # Enforce stability via radius/angle parameterization
        radius = torch.sigmoid(raw[:, :, 3]) * self.MAX_RADIUS
        angle = torch.sigmoid(raw[:, :, 4]) * math.pi
        a1 = -2.0 * radius * torch.cos(angle)
        a2 = radius * radius

        coeffs = torch.stack([b0, b1, b2, a1, a2], dim=-1)  # (B, N, 5)

        if squeeze:
            coeffs = coeffs.squeeze(0)
        return coeffs
