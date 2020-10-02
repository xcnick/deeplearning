import torch
import torch.nn as nn

from .config import ConfigBase
from .transformer import Encoder
from .bert import BertEmbeddings
from .utils import get_activation, PreTrainedModel


class ElectraConfig(ConfigBase):
    def __init__(
        self,
        vocab_size=30522,
        embedding_size=128,
        hidden_size=256,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class ElectraEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)


class ElectraModel(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.embeddings = ElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        self.encoder = Encoder(
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            config.hidden_act,
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify input_ids")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device=input_ids.device
        )

        embeddings = self.embeddings(input_ids, token_type_ids, position_ids)

        if hasattr(self, "embeddings_project"):
            embeddings = self.embeddings_project(embeddings)

        encoder_outputs, attention_outputs = self.encoder(embeddings, extended_attention_mask)
        outputs = (encoder_outputs,)
        return outputs


class ElectraDiscriminatorPredictions(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.dense(hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()

        return logits


class ElectraForPreTraining(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        logits = self.electra(input_ids, attention_mask, token_type_ids, position_ids)
        outputs = logits

        return outputs


def load_tf_weights_in_electra(model, config, tf_checkpoint_path):
    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        raise ImportError("cannot import tensorflow, please install tensorflow first")

    tf_model = tf.train.load_checkpoint(tf_checkpoint_path)

    def _load_tf_variable(key):
        return tf_model.get_tensor(key).squeeze()

    def _load_torch_weight(param, data):
        assert param.shape == data.shape
        param.data = torch.from_numpy(data)

    def _load_embedding(embedding, embedding_path):
        embedding_weight = _load_tf_variable(embedding_path)
        _load_torch_weight(embedding.weight, embedding_weight)

    def _load_layer_norm(layer_norm, layer_norm_base):
        layer_norm_gamma = _load_tf_variable(f"{layer_norm_base}/gamma")
        layer_norm_beta = _load_tf_variable(f"{layer_norm_base}/beta")
        _load_torch_weight(layer_norm.weight, layer_norm_gamma)
        _load_torch_weight(layer_norm.bias, layer_norm_beta)

    def _load_linear(linear, linear_path):
        linear_weight = _load_tf_variable(f"{linear_path}/kernel")
        linear_weight = np.transpose(linear_weight)
        linear_bias = _load_tf_variable(f"{linear_path}/bias")
        _load_torch_weight(linear.weight, linear_weight)
        _load_torch_weight(linear.bias, linear_bias)

    def _load_self_attention(attention, attention_path):
        query_weight = _load_tf_variable(f"{attention_path}/self/query/kernel")
        key_weight = _load_tf_variable(f"{attention_path}/self/key/kernel")
        value_weight = _load_tf_variable(f"{attention_path}/self/value/kernel")

        query_weight = np.transpose(query_weight)
        key_weight = np.transpose(key_weight)
        value_weight = np.transpose(value_weight)

        query_bias = _load_tf_variable(f"{attention_path}/self/query/bias")
        key_bias = _load_tf_variable(f"{attention_path}/self/key/bias")
        value_bias = _load_tf_variable(f"{attention_path}/self/value/bias")

        _load_torch_weight(attention.query.weight, query_weight)
        _load_torch_weight(attention.key.weight, key_weight)
        _load_torch_weight(attention.value.weight, value_weight)

        _load_torch_weight(attention.query.bias, query_bias)
        _load_torch_weight(attention.key.bias, key_bias)
        _load_torch_weight(attention.value.bias, value_bias)

    # loading embedding layer
    _load_embedding(
        model.electra.embeddings.token_embeddings, "electra/embeddings/word_embeddings",
    )
    _load_embedding(
        model.electra.embeddings.token_type_embeddings, "electra/embeddings/token_type_embeddings",
    )
    _load_embedding(
        model.electra.embeddings.position_embeddings, "electra/embeddings/position_embeddings",
    )
    _load_layer_norm(model.electra.embeddings.layer_norm, "electra/embeddings/LayerNorm")

    # embeddings_project
    if config.embedding_size != config.hidden_size:
        _load_linear(model.electra.embeddings_project, "electra/embeddings_project")

    # loading transformer encoders
    for layer_idx in range(config.num_hidden_layers):
        encoder_layer = model.electra.encoder.layers[layer_idx]
        encoder_path = f"electra/encoder/layer_{layer_idx}"

        _load_self_attention(encoder_layer.self, f"{encoder_path}/attention")
        _load_linear(encoder_layer.self.dense, f"{encoder_path}/attention/output/dense")
        _load_layer_norm(
            encoder_layer.add_norm[0].layer_norm, f"{encoder_path}/attention/output/LayerNorm",
        )

        _load_linear(
            encoder_layer.feed_forward.intermediate, f"{encoder_path}/intermediate/dense",
        )
        _load_linear(encoder_layer.feed_forward.output, f"{encoder_path}/output/dense")
        _load_layer_norm(
            encoder_layer.add_norm[1].layer_norm, f"{encoder_path}/output/LayerNorm",
        )


def load_huggingface_weights_in_electra(model, config, pytorch_model_path):
    try:
        from transformers import ElectraForPreTraining
    except ImportError:
        raise ImportError("cannot import transformers, please install transformers first")

    hf_model = ElectraForPreTraining.from_pretrained(pytorch_model_path)
    model.electra.embeddings.token_embeddings.weight = (
        hf_model.electra.embeddings.word_embeddings.weight
    )

    model.electra.embeddings.position_embeddings.weight = (
        hf_model.electra.embeddings.position_embeddings.weight
    )
    model.electra.embeddings.token_type_embeddings.weight = (
        hf_model.electra.embeddings.token_type_embeddings.weight
    )
    model.electra.embeddings.layer_norm.weight = hf_model.electra.embeddings.LayerNorm.weight
    model.electra.embeddings.layer_norm.bias = hf_model.electra.embeddings.LayerNorm.bias

    if config.embedding_size != config.hidden_size:
        model.electra.embeddings_project.weight = hf_model.electra.embeddings_project.weight

    for layer_idx in range(config.num_hidden_layers):
        layer = model.electra.encoder.layers[layer_idx]
        hf_layer = hf_model.electra.encoder.layer[layer_idx]

        layer.self.query.weight = hf_layer.attention.self.query.weight
        layer.self.query.bias = hf_layer.attention.self.query.bias

        layer.self.key.weight = hf_layer.attention.self.key.weight
        layer.self.key.bias = hf_layer.attention.self.key.bias

        layer.self.value.weight = hf_layer.attention.self.value.weight
        layer.self.value.bias = hf_layer.attention.self.value.bias

        layer.self.dense.weight = hf_layer.attention.output.dense.weight
        layer.self.dense.bias = hf_layer.attention.output.dense.bias

        layer.feed_forward.intermediate.weight = hf_layer.intermediate.dense.weight
        layer.feed_forward.intermediate.bias = hf_layer.intermediate.dense.bias

        layer.feed_forward.output.weight = hf_layer.output.dense.weight
        layer.feed_forward.output.bias = hf_layer.output.dense.bias

        layer.add_norm[0].layer_norm.weight = hf_layer.attention.output.LayerNorm.weight
        layer.add_norm[0].layer_norm.bias = hf_layer.attention.output.LayerNorm.bias
        layer.add_norm[1].layer_norm.weight = hf_layer.output.LayerNorm.weight
        layer.add_norm[1].layer_norm.bias = hf_layer.output.LayerNorm.bias
