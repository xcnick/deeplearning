import math
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable, Dict, Any

from transformer import builder

logger = logging.getLogger(__name__)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _gelu_python(x: torch.Tensor) -> torch.Tensor:
    """ Original Implementation of the gelu activation function
        in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
            (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """ Implementation of the gelu activation function currently
        in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu
    gelu_new = torch.jit.script(gelu_new)

ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": F.tanh,
    "gelu_new": gelu_new,
}


def get_activation(activation_string: str) -> Callable[[torch.Tensor], Any]:
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
            activation_string, list(ACT2FN.keys())))


class PreTrainedModel(nn.Module):

    def __init__(self, config: Union[Dict[str, Any], Callable[..., None]],
                 **kwargs):
        super().__init__()
        if isinstance(config, Dict):
            config = builder.build_config(config)
        self.config = config
        self.weights_name = "model_pt.bin"

    @classmethod
    def from_pretrained(cls, config: Callable[..., None], model_path: str,
                        **kwargs) -> "PreTrainedModel":
        model = cls(config, **kwargs)
        if not os.path.isfile(model_path):
            raise f"Error no file named {model_path} found"

        missing_keys, unexpected_keys = model.load_state_dict(
            torch.load(model_path, map_location="cpu"), strict=False)
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
        """ Save a model and to a directory
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
        output_model_file = os.path.join(save_directory, self.weights_name)

        torch.save(model_to_save.state_dict(), output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
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
