import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

import numpy as np
from mindspore import Tensor

from .transformer_ms import Encoder
from .config import ConfigBase
from .utils_ms import MSPreTrainedModel, get_activation
from typing import Tuple, Optional, Callable


class BertConfig(ConfigBase):
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 16,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
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


class MSBertEmbeddings(nn.Cell):
    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_ids = Tensor(
            np.arange(config.max_position_embeddings)
            .reshape(-1, config.max_position_embeddings)
            .astype(np.int32)
        )
        self.token_type_ids = Tensor(
            np.zeros(config.max_position_embeddings)
            .reshape(-1, config.max_position_embeddings)
            .astype(np.int32)
        )

    def construct(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        input_shape = input_ids.shape
        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :seq_length]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        input_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MSBertPooler(nn.Cell):
    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: Tensor) -> Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MSBertModel(MSPreTrainedModel):
    def __init__(self, config: Callable[..., None], **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.embeddings = MSBertEmbeddings(config)
        self.encoder = Encoder(
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            config.hidden_act,
        )
        self.pooler = MSBertPooler(config)

    def construct(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, ...]:
        input_shape = None
        if input_ids is not None:
            input_shape = input_ids.shape

        if attention_mask is None:
            attention_mask = ops.Ones()(input_shape, mindspore.int32)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        embeddings = self.embeddings(input_ids, token_type_ids, position_ids)

        encoder_outputs, attention_outputs = self.encoder(embeddings, extended_attention_mask)
        pooled_outputs = self.pooler(encoder_outputs)
        outputs = (
            encoder_outputs,
            pooled_outputs,
        )

        return outputs


class MSBertPredictionHeadTransform(nn.Cell):
    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.transform_act_fn = get_activation(config.hidden_act)
        self.layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)

    def construct(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class MSBertLMPredictionHead(nn.Cell):
    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.transform = MSBertPredictionHeadTransform(config)
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size)

    def construct(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MSBertOnlyMLMHead(nn.Cell):
    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.predictions = MSBertLMPredictionHead(config)

    def construct(self, sequence_output: Tensor) -> Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class MSBertForPreTraining(MSPreTrainedModel):
    def __init__(self, config: Callable[..., None], **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.bert = MSBertModel(config)
        self.cls = MSBertOnlyMLMHead(config)

    def construct(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, ...]:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        sequence_output, pooled_output = outputs
        prediction_scores = self.cls(sequence_output)

        outputs = outputs + (prediction_scores,)

        return outputs


def load_tf_weights_in_bert_to_ms(
    model: "nn.Cell", config: Callable[..., None], tf_checkpoint_path: str, with_mlm: bool = True,
) -> None:
    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        raise ImportError("cannot import tensorflow, please install tensorflow first")

    tf_model = tf.train.load_checkpoint(tf_checkpoint_path)

    def _load_tf_variable(key: str) -> np.ndarray:
        return tf_model.get_tensor(key).squeeze()

    def _load_ms_weight(param: mindspore.Parameter, data: np.ndarray) -> None:
        assert param.shape == data.shape
        param.set_data(Tensor(data, mindspore.float32))

    def _load_embedding(embedding: "nn.Cell", embedding_path: str) -> None:
        embedding_weight = _load_tf_variable(embedding_path)
        _load_ms_weight(embedding.embedding_table, embedding_weight)

    def _load_layer_norm(layer_norm: "nn.Cell", layer_norm_base: str) -> None:
        layer_norm_gamma = _load_tf_variable(f"{layer_norm_base}/gamma")
        layer_norm_beta = _load_tf_variable(f"{layer_norm_base}/beta")
        _load_ms_weight(layer_norm.gamma, layer_norm_gamma)
        _load_ms_weight(layer_norm.beta, layer_norm_beta)

    def _load_linear(linear: "nn.Cell", linear_path: str) -> None:
        linear_weight = _load_tf_variable(f"{linear_path}/kernel")
        linear_weight = np.transpose(linear_weight)
        linear_bias = _load_tf_variable(f"{linear_path}/bias")
        _load_ms_weight(linear.weight, linear_weight)
        _load_ms_weight(linear.bias, linear_bias)

    def _load_self_attention(attention: "nn.Cell", attention_path: str) -> None:
        query_weight = _load_tf_variable(f"{attention_path}/self/query/kernel")
        key_weight = _load_tf_variable(f"{attention_path}/self/key/kernel")
        value_weight = _load_tf_variable(f"{attention_path}/self/value/kernel")

        query_weight = np.transpose(query_weight)
        key_weight = np.transpose(key_weight)
        value_weight = np.transpose(value_weight)

        query_bias = _load_tf_variable(f"{attention_path}/self/query/bias")
        key_bias = _load_tf_variable(f"{attention_path}/self/key/bias")
        value_bias = _load_tf_variable(f"{attention_path}/self/value/bias")

        _load_ms_weight(attention.query.weight, query_weight)
        _load_ms_weight(attention.key.weight, key_weight)
        _load_ms_weight(attention.value.weight, value_weight)

        _load_ms_weight(attention.query.bias, query_bias)
        _load_ms_weight(attention.key.bias, key_bias)
        _load_ms_weight(attention.value.bias, value_bias)

    # loading embedding layer
    _load_embedding(
        model.bert.embeddings.token_embeddings, "bert/embeddings/word_embeddings",
    )
    _load_embedding(
        model.bert.embeddings.token_type_embeddings, "bert/embeddings/token_type_embeddings",
    )
    _load_embedding(
        model.bert.embeddings.position_embeddings, "bert/embeddings/position_embeddings",
    )
    _load_layer_norm(model.bert.embeddings.layer_norm, "bert/embeddings/LayerNorm")

    # loading transformer encoders
    for layer_idx in range(config.num_hidden_layers):
        encoder_layer = model.bert.encoder.layers[layer_idx]
        encoder_path = f"bert/encoder/layer_{layer_idx}"

        _load_self_attention(encoder_layer.self_attn, f"{encoder_path}/attention")
        _load_linear(encoder_layer.self_attn.dense, f"{encoder_path}/attention/output/dense")
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

    _load_linear(model.bert.pooler.dense, "bert/pooler/dense")

    # MLM
    if not with_mlm:
        return

    embedding_weight = _load_tf_variable("bert/embeddings/word_embeddings")
    _load_ms_weight(model.cls.predictions.decoder.weight, embedding_weight)

    output_bias = _load_tf_variable("cls/predictions/output_bias")
    _load_ms_weight(model.cls.predictions.decoder.bias, output_bias)

    _load_linear(model.cls.predictions.transform.dense, "cls/predictions/transform/dense")
    _load_layer_norm(
        model.cls.predictions.transform.layer_norm, "cls/predictions/transform/LayerNorm",
    )


def load_huggingface_weights_in_bert_to_ms(
    model: "nn.Cell", config: Callable[..., None], pytorch_model_path: str, with_mlm: bool = True,
) -> None:
    try:
        from transformers import BertForPreTraining
        import torch
    except ImportError:
        raise ImportError("cannot import transformers, please install transformers first")

    def _load_ms_weight(param: mindspore.Parameter, data: np.ndarray) -> None:
        assert param.shape == data.shape
        param.set_data(Tensor(data, mindspore.float32))

    def _load_layer_norm(ms_layer_norm: "nn.Cell", torch_layer_norm: "torch.nn.LayerNorm") -> None:
        _load_ms_weight(ms_layer_norm.gamma, torch_layer_norm.weight.detach().numpy())
        _load_ms_weight(ms_layer_norm.beta, torch_layer_norm.bias.detach().numpy())

    def _load_linear(linear: "nn.Cell", torch_linear: "torch.nn.Linear") -> None:
        _load_ms_weight(linear.weight, torch_linear.weight.detach().numpy())
        _load_ms_weight(linear.bias, torch_linear.bias.detach().numpy())

    hf_model = BertForPreTraining.from_pretrained(pytorch_model_path)
    hf_model.eval()

    _load_ms_weight(
        model.bert.embeddings.token_embeddings.embedding_table,
        hf_model.bert.embeddings.word_embeddings.weight.detach().numpy(),
    )
    _load_ms_weight(
        model.bert.embeddings.position_embeddings.embedding_table,
        hf_model.bert.embeddings.position_embeddings.weight.detach().numpy(),
    )
    _load_ms_weight(
        model.bert.embeddings.token_type_embeddings.embedding_table,
        hf_model.bert.embeddings.token_type_embeddings.weight.detach().numpy(),
    )

    _load_layer_norm(model.bert.embeddings.layer_norm, hf_model.bert.embeddings.LayerNorm)

    for layer_idx in range(config.num_hidden_layers):
        layer = model.bert.encoder.layers[layer_idx]
        hf_layer = hf_model.bert.encoder.layer[layer_idx]

        _load_linear(layer.self_attn.query, hf_layer.attention.self.query)
        _load_linear(layer.self_attn.key, hf_layer.attention.self.key)
        _load_linear(layer.self_attn.value, hf_layer.attention.self.value)

        _load_linear(layer.self_attn.dense, hf_layer.attention.output.dense)
        _load_linear(layer.feed_forward.intermediate, hf_layer.intermediate.dense)
        _load_linear(layer.feed_forward.output, hf_layer.output.dense)

        _load_layer_norm(layer.add_norm[0].layer_norm, hf_layer.attention.output.LayerNorm)
        _load_layer_norm(layer.add_norm[1].layer_norm, hf_layer.output.LayerNorm)

    _load_linear(model.bert.pooler.dense, hf_model.bert.pooler.dense)

    # mlm
    if not with_mlm:
        return

    _load_ms_weight(
        model.cls.predictions.decoder.weight,
        hf_model.bert.embeddings.word_embeddings.weight.detach().numpy(),
    )
    _load_ms_weight(
        model.cls.predictions.decoder.bias, hf_model.cls.predictions.decoder.bias.detach().numpy(),
    )

    _load_linear(model.cls.predictions.transform.dense, hf_model.cls.predictions.transform.dense)

    _load_layer_norm(
        model.cls.predictions.transform.layer_norm, hf_model.cls.predictions.transform.LayerNorm
    )
