import torch
import torch.nn as nn

from .config import ConfigBase
from .transformer import Encoder
from .bert import BertEmbeddings, BertPooler, BertModel
from .utils import get_activation, PreTrainedModel


class AlbertConfig(ConfigBase):
    def __init__(
        self,
        vocab_size=30000,
        embedding_size=128,
        hidden_size=4096,
        num_hidden_layers=12,
        num_hidden_groups=1,
        num_attention_heads=64,
        intermediate_size=16384,
        inner_group_num=1,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
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
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.inner_group_num = inner_group_num
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class AlbertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)

        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=1e-12)


class AlbertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.layer_groups = nn.ModuleList(
            [
                Encoder(
                    config.inner_group_num,
                    config.hidden_size,
                    config.num_attention_heads,
                    config.intermediate_size,
                    config.attention_probs_dropout_prob,
                    config.hidden_dropout_prob,
                    config.hidden_act,
                )
                for _ in range(config.num_hidden_groups)
            ]
        )

    def forward(self, x, mask=None):
        x = self.embedding_hidden_mapping_in(x)

        for layers_idx in range(self.config.num_hidden_layers):
            group_idx = int(
                layers_idx / (self.config.num_hidden_layers / self.config.num_hidden_groups)
            )
            x, attn = self.layer_groups[group_idx](x, mask)
        return x, attn


class AlbertPooler(BertPooler):
    def __init__(self, config):
        super().__init__(config)


class AlbertModel(BertModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        self.pooler = AlbertPooler(config)


class AlbertForPreTraining(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.albert = AlbertModel(config)
        self.cls = AlbertOnlyMLMHead(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        sequence_output, pooled_output = outputs
        prediction_scores = self.cls(sequence_output)

        outputs = outputs + (prediction_scores,)

        return outputs


# class AlbertMLMHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.LayerNorm = nn.LayerNorm(config.embedding_size)
#         self.bias = nn.Parameter(torch.zeros(config.vocab_size))
#         self.dense = nn.Linear(config.hidden_size, config.embedding_size)
#         self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
#         self.activation = ACT2FN[config.hidden_act]

#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.activation(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
#         hidden_states = self.decoder(hidden_states)

#         prediction_scores = hidden_states

#         return prediction_scores


class AlbertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.transform_act_fn = get_activation(config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class AlbertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = AlbertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class AlbertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = AlbertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


def load_tf_weights_in_albert(model, config, tf_checkpoint_path, with_mlm=True):
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
        model.albert.embeddings.token_embeddings, "bert/embeddings/word_embeddings",
    )
    _load_embedding(
        model.albert.embeddings.token_type_embeddings, "bert/embeddings/token_type_embeddings",
    )
    _load_embedding(
        model.albert.embeddings.position_embeddings, "bert/embeddings/position_embeddings",
    )
    _load_layer_norm(model.albert.embeddings.layer_norm, "bert/embeddings/LayerNorm")

    _load_linear(
        model.albert.encoder.embedding_hidden_mapping_in,
        "bert/encoder/embedding_hidden_mapping_in",
    )
    for group_idx in range(config.num_hidden_groups):
        for inner_idx in range(config.inner_group_num):
            encoder_layer = model.albert.encoder.layer_groups[group_idx].layers[inner_idx]
            encoder_path = f"bert/encoder/transformer/group_{group_idx}/inner_group_{inner_idx}"

            _load_self_attention(encoder_layer.self, f"{encoder_path}/attention_1")
            _load_linear(
                encoder_layer.self.dense, f"{encoder_path}/attention_1/output/dense",
            )

            _load_layer_norm(
                encoder_layer.add_norm[0].layer_norm, f"{encoder_path}/LayerNorm",
            )

            _load_linear(
                encoder_layer.feed_forward.intermediate, f"{encoder_path}/ffn_1/intermediate/dense",
            )
            _load_linear(
                encoder_layer.feed_forward.output,
                f"{encoder_path}/ffn_1/intermediate/output/dense",
            )
            _load_layer_norm(
                encoder_layer.add_norm[1].layer_norm, f"{encoder_path}/LayerNorm_1",
            )

    _load_linear(model.albert.pooler.dense, "bert/pooler/dense")

    # MLM
    if not with_mlm:
        return

    _load_embedding(model.cls.predictions.decoder, "bert/embeddings/word_embeddings")
    output_bias = _load_tf_variable("cls/predictions/output_bias")
    _load_torch_weight(model.cls.predictions.decoder.bias, output_bias)

    _load_linear(model.cls.predictions.transform.dense, "cls/predictions/transform/dense")
    _load_layer_norm(
        model.cls.predictions.transform.layer_norm, "cls/predictions/transform/LayerNorm",
    )


def load_huggingface_weights_in_albert(model, config, pytorch_model_path, with_mlm=True):
    try:
        from transformers import AlbertForMaskedLM
    except ImportError:
        raise ImportError("cannot import transformers, please install transformers first")

    hf_model = AlbertForMaskedLM.from_pretrained(pytorch_model_path)

    model.albert.embeddings.token_embeddings.weight = (
        hf_model.albert.embeddings.word_embeddings.weight
    )

    model.albert.embeddings.position_embeddings.weight = (
        hf_model.albert.embeddings.position_embeddings.weight
    )
    model.albert.embeddings.token_type_embeddings.weight = (
        hf_model.albert.embeddings.token_type_embeddings.weight
    )
    model.albert.embeddings.layer_norm.weight = hf_model.albert.embeddings.LayerNorm.weight
    model.albert.embeddings.layer_norm.bias = hf_model.albert.embeddings.LayerNorm.bias

    model.albert.encoder.embedding_hidden_mapping_in.weight = (
        hf_model.albert.encoder.embedding_hidden_mapping_in.weight
    )
    model.albert.encoder.embedding_hidden_mapping_in.bias = (
        hf_model.albert.encoder.embedding_hidden_mapping_in.bias
    )
    for group_idx in range(config.num_hidden_groups):
        for inner_idx in range(config.inner_group_num):
            encoder_layer = model.albert.encoder.layer_groups[group_idx].layers[inner_idx]
            hf_encoder_layer = hf_model.albert.encoder.albert_layer_groups[group_idx].albert_layers[
                inner_idx
            ]

            encoder_layer.self.query.weight = hf_encoder_layer.attention.query.weight
            encoder_layer.self.query.bias = hf_encoder_layer.attention.query.bias

            encoder_layer.self.key.weight = hf_encoder_layer.attention.key.weight
            encoder_layer.self.key.bias = hf_encoder_layer.attention.key.bias

            encoder_layer.self.value.weight = hf_encoder_layer.attention.value.weight
            encoder_layer.self.value.bias = hf_encoder_layer.attention.value.bias

            encoder_layer.self.dense.weight = hf_encoder_layer.attention.dense.weight
            encoder_layer.self.dense.bias = hf_encoder_layer.attention.dense.bias

            encoder_layer.feed_forward.intermediate.weight = hf_encoder_layer.ffn.weight
            encoder_layer.feed_forward.intermediate.bias = hf_encoder_layer.ffn.bias

            encoder_layer.feed_forward.output.weight = hf_encoder_layer.ffn_output.weight
            encoder_layer.feed_forward.output.bias = hf_encoder_layer.ffn_output.bias

            encoder_layer.add_norm[
                0
            ].layer_norm.weight = hf_encoder_layer.attention.LayerNorm.weight
            encoder_layer.add_norm[0].layer_norm.bias = hf_encoder_layer.attention.LayerNorm.bias

            encoder_layer.add_norm[
                1
            ].layer_norm.weight = hf_encoder_layer.full_layer_layer_norm.weight
            encoder_layer.add_norm[1].layer_norm.bias = hf_encoder_layer.full_layer_layer_norm.bias

    model.albert.pooler.dense.weight = hf_model.albert.pooler.weight
    model.albert.pooler.dense.bias = hf_model.albert.pooler.bias

    # mlm
    if not with_mlm:
        return

    model.cls.predictions.decoder.weight = hf_model.albert.embeddings.word_embeddings.weight
    model.cls.predictions.decoder.bias = hf_model.predictions.decoder.bias
    model.cls.predictions.transform.dense.weight = hf_model.predictions.dense.weight
    model.cls.predictions.transform.dense.bias = hf_model.predictions.dense.bias
    model.cls.predictions.transform.layer_norm.weight = hf_model.predictions.LayerNorm.weight
    model.cls.predictions.transform.layer_norm.bias = hf_model.predictions.LayerNorm.bias
