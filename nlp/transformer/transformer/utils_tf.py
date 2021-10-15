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
        raise KeyError(
            "function {} not found in ACT2FN mapping {}".format(
                activation_string, list(ACT2FN.keys())
            )
        )


class TFPreTrainedModel(tf.keras.Model):
    def __init__(self, config: Union[Dict[str, Any], Callable[..., None]], **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(config, Dict):
            config = builder.build_config(config)
        self.config = config
        self.weights_name = "model_tf.bin"

    @classmethod
    def from_pretrained(
        cls, config: Callable[..., None], model_path: str, **kwargs
    ) -> "TFPreTrainedModel":
        model = cls(config, **kwargs)

        inputs = {
            "input_ids": tf.zeros((1, 1), dtype=tf.int32),
        }
        model(inputs)

        model.load_weights(model_path)
        # state_dict = torch.load(model_path, map_location="cpu")

        # # copy state_dict so _load_from_state_dict can modify it
        # metadata = getattr(state_dict, "_metadata", None)
        # state_dict = state_dict.copy()
        # if metadata is not None:
        #     state_dict._metadata = metadata

        # # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # # so we need to apply the function recursively.
        # def load(module: nn.Module, prefix=""):
        #     local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        #     module._load_from_state_dict(
        #         state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
        #     )
        #     for name, child in module._modules.items():
        #         if child is not None:
        #             load(child, prefix + name + ".")

        # # Make sure we are able to load base models as well as derived models (with heads)
        # start_prefix = ""
        # model_to_load = model
        # has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
        # if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
        #     start_prefix = cls.base_model_prefix + "."
        # if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
        #     model_to_load = getattr(model, cls.base_model_prefix)

        # load(model_to_load, prefix=start_prefix)

        # if model.__class__.__name__ != model_to_load.__class__.__name__:
        #     base_model_state_dict = model_to_load.state_dict().keys()
        #     head_model_state_dict_without_base_prefix = [
        #         key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
        #     ]

        #     missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

        # missing_keys, unexpected_keys = model.load_state_dict(
        #     torch.load(model_path, map_location="cpu"), strict=False
        # )
        # if len(missing_keys) > 0:
        #     logger.info(
        #         "Weights of {} not initialized from pretrained model: {}".format(
        #             model.__class__.__name__, missing_keys
        #         )
        #     )
        # if len(unexpected_keys) > 0:
        #     logger.info(
        #         "Weights from pretrained model not used in {}: {}".format(
        #             model.__class__.__name__, unexpected_keys
        #         )
        #     )
        # model.eval()

        return model

    def save_pretrained(self, save_directory: str) -> None:
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

            Arguments:
                save_directory: directory to which to save.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, self.weights_name)

        # torch.save(model_to_save.state_dict(), output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))

    def get_extended_attention_mask(
        self, attention_mask: tf.Tensor, input_shape: Tuple[int, ...]
    ) -> tf.Tensor:
        if tf.ndim(attention_mask) == 3:
            extended_attention_mask = attention_mask[:, tf.newaxis, :, :]
        elif tf.ndim(attention_mask) == 2:
            extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        #extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask
