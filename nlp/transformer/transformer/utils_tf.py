import os
import logging
import tensorflow as tf
from typing import Tuple, Callable, Dict, Union, Any

from transformer import builder

logger = logging.getLogger(__name__)

ACT2FN = {
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.activations.swish,
    "gelu": lambda x: tf.keras.activations.gelu(x, approximate=False),
    "tanh": tf.keras.activations.tanh,
    "gelu_new": lambda x: tf.keras.activations.gelu(x, approximate=True),
}


def get_activation(activation_string: str) -> Callable[[tf.Tensor], Any]:
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
            activation_string, list(ACT2FN.keys())))


class TFPreTrainedModel(tf.keras.Model):

    def __init__(self, config: Union[Dict[str, Any], Callable[..., None]],
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(config, Dict):
            config = builder.build_config(config)
        self.config = config
        self.weights_name = "model_tf.bin"

    @classmethod
    def from_pretrained(cls, config: Callable[..., None], model_path: str,
                        **kwargs) -> "TFPreTrainedModel":
        model = cls(config, **kwargs)

        inputs = {
            "input_ids": tf.zeros((1, 1), dtype=tf.int32),
        }
        model(inputs)

        model.load_weights(model_path)

        return model

    def save_pretrained(self, save_directory: str) -> None:
        """ Save a model to a directory
            Arguments:
                save_directory: directory to which to save.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model can be saved"

        # Only save the model itself if we are using distributed training
        # model_to_save = self.module if hasattr(self, "module") else self

        # If we save using the predefined names,
        # we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, self.weights_name)

        # torch.save(model_to_save.state_dict(), output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))

    def get_extended_attention_mask(self, attention_mask: tf.Tensor,
                                    input_shape: Tuple[int, ...]) -> tf.Tensor:
        if tf.ndim(attention_mask) == 3:
            extended_attention_mask = attention_mask[:, tf.newaxis, :, :]
        elif tf.ndim(attention_mask) == 2:
            extended_attention_mask = attention_mask[:, tf.newaxis,
                                                     tf.newaxis, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) "
                             "or attention_mask (shape {})".format(
                                 input_shape, attention_mask.shape))

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask
