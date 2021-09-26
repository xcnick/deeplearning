import tensorflow as tf
import numpy as np

from transformer.utils.logging import get_logger, print_log
from transformer.builder import TF_CALLBACKS
from .utils import round_float_to_str


@TF_CALLBACKS.register_module()
class TextLogger(tf.keras.callbacks.Callback):
    def __init__(
        self,
        print_freq: int = 50,
        print_summary: bool = True,
        initial_epoch: int = 0,
        epochs: int = None,
        steps_per_epoch: int = None,
        val_steps: int = None,
    ) -> None:
        super().__init__()

        self.print_freq = print_freq
        self.print_summary = print_summary
        self.global_epoch = initial_epoch
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.val_steps = val_steps

        self.logger = get_logger(name="transformer")

        self.logs_history = {}
        self.val_logs_history = {}

    def on_train_begin(self, logs=None):
        if self.print_summary:
            summary_log = (
                f"Train for {self.epochs} epochs with {self.steps_per_epoch} "
                f"steps per epoch, validate for {self.val_steps} steps"
            )
            print_log(summary_log, self.logger)

        self.logs_history = {k: None for k in self.model.metrics_names}
        self.logs_history.update({"lr": None})

    def on_epoch_begin(self, epoch, logs=None):
        self.global_epoch += 1
        # reset log history
        self.logs_history = {k: None for k in self.logs_history}

    def on_train_batch_end(self, batch, logs=None):
        batch += 1
        self._update_logs(logs=logs)
        if batch % self.print_freq == 0 or batch == self.steps_per_epoch:
            base_log = f"Epoch [{self.global_epoch}/{self.epochs}][{batch}/{self.steps_per_epoch}]"
            metric_log = []
            for key_, value_ in self.logs_history.items():
                if value_ is None:
                    continue
                print_value_ = round_float_to_str(value_)
                metric_log.append(f"{key_}: {print_value_}")
            metric_log = ", ".join(metric_log)

            log = f"{base_log} {metric_log}"
            print_log(log, self.logger)

    def on_test_begin(self, logs=None):
        if self.print_summary:
            summary_log = "Evaluate for {} steps".format(self.val_steps)
            print_log(summary_log, self.logger)

        self.val_logs_history = {}

    def on_test_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        # if self._is_fit:
        logs = {"val_" + k: v for k, v in logs.items()}
        self._update_val_logs(logs=logs)

    def on_test_end(self, logs=None):
        if logs is None:
            logs = {}
        # if self._is_fit:
        logs = {"val_" + k: v for k, v in logs.items()}
        self._update_val_logs(logs=logs)
        self.val_logs_history = {
            key_: np.mean(value_) for key_, value_ in self.val_logs_history.items()
        }
        # we only print evaluation logs in `on_test_end` when calling `model.evaluate()`,
        # otherwise (calling `model.fit()`) the evaluation logs will be updated to epoch log
        # and print in `on_epoch_end`.
        # if not self._is_fit:
        metric_log = []
        for key_, value_ in self.val_logs_history.items():
            print_value_ = round_float_to_str(value_)
            metric_log.append("{}: {}".format(key_, print_value_))
        metric_log = ", ".join(metric_log)
        if len(metric_log) > 0:
            print_log(metric_log, self.logger)

    def on_epoch_end(self, epoch, logs=None):
        self._update_logs(logs=logs)
        self._update_logs(logs=self.val_logs_history)
        base_log = f"Epoch [{self.global_epoch}/{self.epochs}]"
        metric_log = []
        for key_, value_ in self.logs_history.items():
            if value_ is None:
                continue
            print_value_ = round_float_to_str(value_)
            metric_log.append(f"{key_}: {print_value_}")
        metric_log = ", ".join(metric_log)

        log = f"{base_log} {metric_log}"
        print_log(log, self.logger)

    def _update_logs(self, logs=None):
        if logs is None:
            logs = {}
        self.logs_history.update(logs)

    def _update_val_logs(self, logs=None):
        if logs is None:
            logs = {}
        self.val_logs_history.update(logs)


@TF_CALLBACKS.register_module()
class TensorBoardLogger(tf.keras.callbacks.TensorBoard):
    """TensorBoard logger."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
