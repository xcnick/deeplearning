import tensorflow as tf
import numpy as np
import logging

from transformer.utils.logging import get_logger, print_log
from transformer.builder import TF_CALLBACKS
from .utils import round_float_to_str


@TF_CALLBACKS.register_module()
class PolyLRScheduler(tf.keras.callbacks.Callback):
    """Polynomial learning rate schedule.

    Args:
        total_steps: int, training total steps.
        base_lr: float, base learning rate.
        warmup_iters: int, number of warm-up iterations.
        warmup_lr: float, initial learning rate.
        power: float, new_lr = base_lr * (1 - global_step / total_steps)**power.
        min_lr: float, minimum learning rate.
        optimizer: str, the name of optimizer that using this learning rate scheduler,
            default to apply to all optimizers.
    """

    def __init__(
        self,
        total_steps,
        base_lr,
        warmup_iters=0,
        warmup_lr=None,
        power=0.9,
        min_lr=None,
        optimizer=None,
    ):
        super().__init__()

        if warmup_lr is None:
            warmup_lr = base_lr

        self.total_steps = total_steps
        self.base_lr = base_lr
        self.warmup_iters = warmup_iters
        self.warmup_lr = warmup_lr
        self.power = power
        self.min_lr = min_lr
        self.optimizer = optimizer

        self.logger = get_logger(name="transformer")

        if self.warmup_iters > 0 and self.warmup_lr == self.base_lr:
            print_lr_ = round_float_to_str(self.base_lr)
            print_log(
                "`PolyLRScheduler` got warmup learning rate is equal to base learning "
                f"rate ({print_lr_}), there is no need for warmup, set warmup iterations {self.warmup_iters} -> 0",
                self.logger,
                level=logging.WARNING,
            )
            self.warmup_iters = 0

        self.global_step = 0
        self.global_epoch = 0

    def on_train_begin(self, logs=None):
        init_lr = self.warmup_lr if self.warmup_iters > 0 else self.base_lr
        self._set_lr(lr=init_lr)

    def on_train_batch_begin(self, batch, logs=None):
        if 0 < self.global_step <= self.warmup_iters:
            lr = (
                self.warmup_lr
                + (self.base_lr - self.warmup_lr) / self.warmup_iters * self.global_step
            )
        else:
            lr = self.base_lr * (1 - self.global_step / self.total_steps) ** self.power
        if self.min_lr is not None:
            lr = max(lr, self.min_lr)
        self._set_lr(lr=lr)
        self.global_step += 1

    def on_train_batch_end(self, batch, logs=None):
        logs = self._log_lr(logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.global_epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        logs = self._log_lr(logs=logs)

    def _set_lr(self, lr):
        self.model.optimizer.lr.assign(lr)

    def _log_lr(self, logs=None):
        if logs is None:
            logs = {}
        if self.optimizer is None:
            lr = self.model.optimizer.lr.numpy()
            logs["lr"] = lr

        return logs
