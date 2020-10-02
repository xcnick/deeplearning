import argparse
import logging

import torch
import tensorflow as tf

from transformer.bert import (
    BertConfig,
    BertForPreTraining,
    load_tf_weights_in_bert,
    load_huggingface_weights_in_bert,
)
from transformer.albert import (
    AlbertConfig,
    AlbertForPreTraining,
    load_tf_weights_in_albert,
    load_huggingface_weights_in_albert,
)
from transformer.electra import (
    ElectraConfig,
    ElectraForPreTraining,
    load_tf_weights_in_electra,
    load_huggingface_weights_in_electra,
)
from transformer.gpt2 import GPT2Config, GPT2Model, load_tf_weights_in_gpt2

from transformer.bert_tf import (
    TFBertForPreTraining,
    load_tf_weights_in_bert_to_tf,
    load_huggingface_weights_in_bert_to_tf,
)

logging.basicConfig(level=logging.INFO)

MODEL_CLASSES = {
    "pt": {
        "bert": (BertConfig, BertForPreTraining),
        "albert": (AlbertConfig, AlbertForPreTraining),
        "electra": (ElectraConfig, ElectraForPreTraining),
        "gpt2": (GPT2Config, GPT2Model),
    },
    "tf": {"bert": (BertConfig, TFBertForPreTraining)},
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
}


def convert_weights(model_type, from_model, from_path, config_path, to_model, dump_path):
    config_class, model_class = MODEL_CLASSES[to_model][model_type]
    config = config_class.from_json_file(config_path)
    model = model_class(config)
    load_weights_fct = LOAD_WEIGHTS_MAPS[to_model][model_type][from_model]
    if to_model == "tf":
        input_ids = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]], dtype=tf.int32)
        model(input_ids)
    load_weights_fct(model, config, from_path)

    if to_model == "pt":
        torch.save(model.state_dict(), dump_path)
    else:
        model.save_weights(dump_path)
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
        "--to_model", type=str, required=True, choices=("tf", "pt"), help="To model : tf and pt",
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
