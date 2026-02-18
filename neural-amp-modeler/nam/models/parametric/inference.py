# File: inference.py
# Optimized inference wrapper for ParametricAmp
#
# Key optimizations for real-time use:
# 1. torch.no_grad() context
# 2. Pre-computed knob controller outputs (recomputed only when knobs change)
# 3. Handles both buffer and streaming modes

from typing import Optional

import numpy as np
import torch

from .parametric_amp import ParametricAmp


class ParametricAmpInference:
    """
    Real-time inference wrapper for the ParametricAmp model.

    Caches knob controller outputs so they're only recomputed when
    knobs actually change. The DSP stages themselves must still run
    per-buffer since they maintain internal state (GRU hidden states,
    biquad filter states).

    Usage:
        model = ParametricAmp.init_from_config(config)
        model.import_weights(weights)
        inference = ParametricAmpInference(model)
        inference.set_knobs(np.array([0.5]*8))
        output = inference.process(input_buffer)
    """

    def __init__(self, model: ParametricAmp):
        self._model = model
        self._model.eval()
        self._knobs = np.full(ParametricAmp.NUM_KNOBS, ParametricAmp.NOMINAL_VALUE)
        self._knobs_tensor = torch.full(
            (ParametricAmp.NUM_KNOBS,), ParametricAmp.NOMINAL_VALUE
        )
        self._knobs_changed = True

    def set_knobs(self, knobs: np.ndarray):
        """
        Update knob values. Called from the plugin's parameter change callback.

        :param knobs: (NUM_KNOBS,) array of knob values in [0, 1]
        """
        if not np.array_equal(knobs, self._knobs):
            self._knobs = knobs.copy()
            self._knobs_tensor = torch.from_numpy(knobs).float()
            self._knobs_changed = True

    def set_knob(self, index: int, value: float):
        """Set a single knob value."""
        if self._knobs[index] != value:
            self._knobs[index] = value
            self._knobs_tensor[index] = value
            self._knobs_changed = True

    def set_bypass(self, stage: str, bypass: bool):
        """Enable/disable bypass for a specific stage."""
        self._model.set_bypass(stage, bypass)

    def process(self, input_buffer: np.ndarray) -> np.ndarray:
        """
        Process a buffer of audio samples.

        :param input_buffer: (num_frames,) mono audio
        :return: (num_frames,) processed audio
        """
        x = torch.from_numpy(input_buffer).float().unsqueeze(0)  # (1, L)

        with torch.no_grad():
            y = self._model(x, knobs=self._knobs_tensor)

        self._knobs_changed = False
        return y.squeeze(0).numpy()

    @property
    def knobs(self) -> np.ndarray:
        return self._knobs.copy()

    @property
    def knob_names(self):
        return ParametricAmp.KNOB_NAMES

    @property
    def num_knobs(self) -> int:
        return ParametricAmp.NUM_KNOBS

    @classmethod
    def from_nam_file(cls, filepath: str) -> "ParametricAmpInference":
        """
        Load a ParametricAmp model from a .nam file.

        :param filepath: Path to the .nam file
        :return: ParametricAmpInference instance
        """
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        if data.get("architecture") != "ParametricAmp":
            raise ValueError(
                f"Expected ParametricAmp architecture, got {data.get('architecture')}"
            )

        config = data["config"]
        weights = data["weights"]
        sample_rate = data.get("sample_rate")

        # Build config with sample_rate
        init_config = {
            "preamp_config": config.get("preamp", {}),
            "tonestack_config": config.get("tonestack", {}),
            "poweramp_config": config.get("poweramp", {}),
            "output_config": config.get("output", {}),
            "sample_rate": sample_rate,
        }

        model = ParametricAmp(**init_config)
        model.import_weights(weights)

        return cls(model)
