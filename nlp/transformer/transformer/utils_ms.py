import os
import logging
from typing import Callable, Dict, Union, Any
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from transformer import builder

logger = logging.getLogger(__name__)

ACT2FN = {
    "relu": ops.ReLU(),
    "swish": ops.HSwish(),
    "gelu": ops.GeLU(),
    "tanh": ops.Tanh(),
}


def get_activation(activation_string: str):
    return ACT2FN[activation_string]


class MSPreTrainedModel(nn.Cell):

    def __init__(self, config: Union[Dict[str, Any], Callable[..., None]],
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(config, Dict):
            config = builder.build_config(config)
        self.config = config
        self.weights_name = "model_ms.ckpt"

    @classmethod
    def from_pretrained(cls, config: Callable[..., None], model_path: str,
                        **kwargs) -> "MSPreTrainedModel":
        model = cls(config, **kwargs)

        param_dict = load_checkpoint(model_path)
        load_param_into_net(model, param_dict)

        return model

    def save_pretrained(self, save_directory):
        """ Save a model file to a directory

            Arguments:
                save_directory: directory to which to save.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model can be saved"

        # Only save the model itself if we are using distributed training
        _ = self.module if hasattr(self, "module") else self

        # If we save using the predefined names,
        # we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, self.weights_name)

        # torch.save(model_to_save.state_dict(), output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))

    def get_extended_attention_mask(self, attention_mask):
        extended_attention_mask = None
        expand_dims = ops.ExpandDims()
        if attention_mask.ndim == 3:
            extended_attention_mask = expand_dims(attention_mask, 1)
        elif attention_mask.ndim == 2:
            attention_mask = expand_dims(attention_mask, 1)
            extended_attention_mask = expand_dims(attention_mask, 1)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask
