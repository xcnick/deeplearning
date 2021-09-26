import tensorflow as tf

from transformer.builder import TF_CALLBACKS


@TF_CALLBACKS.register_module()
class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """TensorBoard logger."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
