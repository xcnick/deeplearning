import os
import logging
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
from typing import Union, Callable, Dict, Any

from transformer import builder

logger = logging.getLogger(__name__)


def swish(x: flow.Tensor) -> flow.Tensor:
    return x * flow.sigmoid(x)


ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": F.gelu,
    "tanh": F.tanh,
}


def get_activation(activation_string: str) -> Callable[[flow.Tensor], Any]:
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
            activation_string, list(ACT2FN.keys())))


class OFPreTrainedModel(nn.Module):

    def __init__(self, config: Union[Dict[str, Any], Callable[..., None]],
                 **kwargs):
        super().__init__()
        if isinstance(config, Dict):
            config = builder.build_config(config)
        self.config = config

    @classmethod
    def from_pretrained(cls, config: Callable[..., None], model_path: str,
                        **kwargs) -> "OFPreTrainedModel":
        model = cls(config, **kwargs)
        if not os.path.isdir(model_path):
            raise f"Error no file named {model_path} found"

        missing_keys, unexpected_keys = model.load_state_dict(
            flow.load(model_path), strict=False)
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".
                format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
        model.eval()

        return model

    def save_pretrained(self, save_directory: str):
        """ Save a model file to a directory
            Arguments:
                save_directory: directory to which to save.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # If we save using the predefined names,
        # we can load using `from_pretrained`
        # output_model_file = os.path.join(save_directory, self.weights_name)

        flow.save(model_to_save.state_dict(), save_directory)

        logger.info("Model weights saved in {}".format(save_directory))

    def get_extended_attention_mask(
        self,
        attention_mask: flow.Tensor,
        input_ids: flow.Tensor,
    ):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) "
                             "or attention_mask (shape {})".format(
                                 input_ids.shape, attention_mask.shape))

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask
