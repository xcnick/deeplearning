import argparse
import logging

import torch
import tensorflow as tf
import mindspore
import oneflow as flow

from transformer.transformer.bert import (
    BertForPreTraining,
    load_tf_weights_in_bert,
    load_huggingface_weights_in_bert,
)
from transformer.transformer.albert import (
    AlbertForPreTraining,
    load_tf_weights_in_albert,
    load_huggingface_weights_in_albert,
)
from transformer.transformer.electra import (
    ElectraForPreTraining,
    load_tf_weights_in_electra,
    load_huggingface_weights_in_electra,
)
from transformer.transformer.gpt2 import GPT2Model, load_tf_weights_in_gpt2
from transformer.transformer.bert_tf import (
    TFBertForPreTraining,
    load_tf_weights_in_bert_to_tf,
    load_huggingface_weights_in_bert_to_tf,
)
from transformer.transformer.bert_ms import (
    MSBertForPreTraining,
    load_tf_weights_in_bert_to_ms,
    load_huggingface_weights_in_bert_to_ms,
)
from transformer.transformer.bert_of import (
    OFBertForPreTraining,
    load_tf_weights_in_bert,
    load_huggingface_weights_in_bert,
)
from transformer.transformer import ConfigBase

logging.basicConfig(level=logging.INFO)

MODEL_CLASSES = {
    "pt": {
        "bert": BertForPreTraining,
        "albert": AlbertForPreTraining,
        "electra": ElectraForPreTraining,
        "gpt2": GPT2Model,
    },
    "tf": {"bert": TFBertForPreTraining},
    "ms": {"bert": MSBertForPreTraining},
    "of": {"bert": OFBertForPreTraining},
}

LOAD_WEIGHTS_MAPS = {
    "pt": {
        "bert": {"tf": load_tf_weights_in_bert, "hf": load_huggingface_weights_in_bert},
        "albert": {"tf": load_tf_weights_in_albert, "hf": load_huggingface_weights_in_albert},
        "electra": {"tf": load_tf_weights_in_electra, "hf": load_huggingface_weights_in_electra},
        "gpt2": {"tf": load_tf_weights_in_gpt2, "hf": None},
    },
    "tf": {
        "bert": {"tf": load_tf_weights_in_bert_to_tf, "hf": load_huggingface_weights_in_bert_to_tf}
    },
    "ms": {
        "bert": {"tf": load_tf_weights_in_bert_to_ms, "hf": load_huggingface_weights_in_bert_to_ms}
    },
    "of": {"bert": {"tf": load_tf_weights_in_bert, "hf": load_huggingface_weights_in_bert}},
}


def convert_weights(
    model_type: str,
    from_model: str,
    from_path: str,
    config_path: str,
    to_model: str,
    dump_path: str,
):
    model_class = MODEL_CLASSES[to_model][model_type]
    config = ConfigBase(config_path)
    model = model_class(config)
    load_weights_fct = LOAD_WEIGHTS_MAPS[to_model][model_type][from_model]
    if to_model == "tf":
        input_ids = tf.ones([3, 4], dtype=tf.int32)
        model(input_ids)
    load_weights_fct(model, config, from_path)

    if to_model == "pt":
        torch.save(model.state_dict(), dump_path)
    elif to_model == "tf":
        model.save_weights(dump_path)
    elif to_model == "ms":
        mindspore.save_checkpoint(model, dump_path)
    elif to_model == "of":
        flow.save(model.state_dict(), dump_path)
    print("Save {} model to {}".format(to_model, dump_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=("bert", "albert", "electra", "gpt2"),
        help="Model type : ['bert', 'albert', 'electra', 'gpt2']",
    )
    parser.add_argument(
        "--from_model",
        type=str,
        required=True,
        choices=("tf", "hf"),
        help="From model : tf and hf",
    )

    parser.add_argument(
        "--from_path", type=str, help="Tensorflow checkpoint / Huggingface model file path."
    )
    parser.add_argument("--config_path", type=str, help="Config file path.")
    parser.add_argument(
        "--to_model",
        type=str,
        required=True,
        choices=("tf", "pt", "ms", "of"),
        help="To model : tensorflow / pytorch / mindspore / oneflow",
    )
    parser.add_argument(
        "--dump_path", default=None, type=str, required=True, help="Output model file path"
    )

    args = parser.parse_args()

    convert_weights(
        args.model_type,
        args.from_model,
        args.from_path,
        args.config_path,
        args.to_model,
        args.dump_path,
    )
