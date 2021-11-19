import os
import logging
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Union, Callable, Dict, Any

from transformer import builder

logger = logging.getLogger(__name__)

ACT2FN = {
    "relu": F.relu,
    "swish": F.swish,
    "gelu": F.gelu,
    "tanh": F.tanh,
}


def get_activation(activation_string: str) -> Callable[[paddle.Tensor], Any]:
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
            activation_string, list(ACT2FN.keys())))


class PDPreTrainedModel(nn.Layer):

    def __init__(self, config: Union[Dict[str, Any], Callable[..., None]],
                 **kwargs):
        super().__init__()
        if isinstance(config, Dict):
            config = builder.build_config(config)
        self.config = config
        self.weights_name = "model_pd.bin"

    @classmethod
    def from_pretrained(cls, config: Callable[..., None], model_path: str,
                        **kwargs) -> "PDPreTrainedModel":
        model = cls(config, **kwargs)
        if not os.path.isfile(model_path):
            raise f"Error no file named {model_path} found"

        model.set_state_dict(paddle.load(model_path))

        model.eval()

        return model

    def save_pretrained(self, save_directory: str):
        """ Save a model to a directory
                save_directory: directory to which to save.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # If we save using the predefined names,
        # we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, self.weights_name)

        paddle.save(model_to_save.state_dict(), output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))

    def get_extended_attention_mask(
        self,
        attention_mask: paddle.Tensor,
        input_ids: paddle.Tensor,
    ) -> paddle.Tensor:
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze((1, 2))
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) "
                             "or attention_mask (shape {})".format(
                                 input_ids.shape, attention_mask.shape))

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask
