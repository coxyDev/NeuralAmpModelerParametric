# File: lightning_module.py
# Extended LightningModule for parametric amp training
#
# Handles the 3-tuple batch format (x, knobs, y) by extracting
# knobs and passing them as a keyword argument to the model.

from typing import Dict, Optional, Tuple

import torch

from ...train.lightning_module import LightningModule, LossConfig, _LossItem


class ParametricLightningModule(LightningModule):
    """
    Extends the base LightningModule to handle parametric amp training.

    Key differences from the base:
    1. Batch format is (x, knobs, y) instead of (x, y)
    2. Knobs are passed as keyword argument to avoid positional arg conflict
       with BaseNet.forward(x, pad_start, **kwargs)
    """

    def _shared_step(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, _LossItem]]:
        """
        Unpack 3-tuple batch and pass knobs as keyword argument.

        :param batch: (x, knobs, y) where
            x: (B, NX+NY-1) input audio
            knobs: (B, NUM_KNOBS) knob values
            y: (B, NY) target output
        """
        x, knobs, targets = batch

        # Pass knobs as keyword arg so BaseNet.forward() receives them
        # in **kwargs and propagates to _forward(x, knobs=knobs)
        preds = self.net(x, pad_start=False, knobs=knobs)

        # Compute all relevant losses (same as base class)
        from ..losses import (
            apply_pre_emphasis_filter as _apply_pre_emphasis_filter,
            esr as _esr,
            multi_resolution_stft_loss as _multi_resolution_stft_loss,
            mse_fft as _mse_fft,
        )

        loss_dict = {}

        # MSE loss
        if self._loss_config.fourier:
            loss_dict["MSE_FFT"] = _LossItem(1.0, _mse_fft(preds, targets))
        else:
            loss_dict["MSE"] = _LossItem(1.0, self._mse_loss(preds, targets))

        # Pre-emphasized MSE
        if self._loss_config.pre_emph_weight is not None:
            loss_dict["Pre-emphasized MSE"] = _LossItem(
                self._loss_config.pre_emph_weight,
                self._mse_loss(
                    preds, targets, pre_emph_coef=self._loss_config.pre_emph_coef
                ),
            )

        # MRSTFT
        if self._loss_config.mrstft_weight is not None:
            loss_dict["MRSTFT"] = _LossItem(
                self._loss_config.mrstft_weight,
                self._mrstft_loss(preds, targets),
            )

        # Pre-emphasized MRSTFT
        if self._loss_config.pre_emph_mrstft_weight is not None:
            loss_dict["Pre-emphasized MRSTFT"] = _LossItem(
                self._loss_config.pre_emph_mrstft_weight,
                self._mrstft_loss(
                    preds,
                    targets,
                    pre_emph_coef=self._loss_config.pre_emph_mrstft_coef,
                ),
            )

        # DC loss
        dc_weight = self._loss_config.dc_weight
        if dc_weight is not None and dc_weight > 0.0:
            import torch.nn as nn

            mean_dims = torch.arange(1, preds.ndim).tolist()
            dc_loss = nn.MSELoss()(
                preds.mean(dim=mean_dims), targets.mean(dim=mean_dims)
            )
            loss_dict["DC MSE"] = _LossItem(dc_weight, dc_loss)

        return preds, targets, loss_dict
