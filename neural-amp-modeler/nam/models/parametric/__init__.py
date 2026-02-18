# File: __init__.py
# Parametric amp modelling module
# Grey-box DDSP hybrid architecture for tunable amp profiling

from .dsp_primitives import (
    BiquadCascade,
    DifferentiableBiquad,
    DifferentiableEnvelopeFollower,
    GRUNonlinearity,
    KnobController,
    ParametricBiquad,
    StableBiquadController,
)
from .inference import ParametricAmpInference
from .parametric_amp import ParametricAmp
from .stages import OutputStage, PowerAmpStage, PreampStage, ToneStackStage
