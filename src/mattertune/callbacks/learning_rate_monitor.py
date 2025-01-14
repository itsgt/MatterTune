from __future__ import annotations

from typing import Literal

import nshconfig as C


class LearningRateMonitorConfig(C.Config):
    logging_interval: Literal["step", "epoch"] | None = None
    """Set to ``'epoch'`` or ``'step'`` to log ``lr`` of all optimizers at the same interval, set to ``None`` to log at individual interval according to the ``interval`` key of each scheduler. Default: ``None``."""

    log_momentum: bool = False
    """Option to also log the momentum values of the optimizer, if the optimizer has the ``momentum`` or ``betas`` attribute. Default: ``False``."""

    log_weight_decay: bool = False
    """Option to also log the weight decay values of the optimizer. Default: ``False``."""

    def create_callback(self):
        from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

        return LearningRateMonitor(
            logging_interval=self.logging_interval,
            log_momentum=self.log_momentum,
            log_weight_decay=self.log_weight_decay,
        )
