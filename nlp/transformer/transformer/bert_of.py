import oneflow as flow
import oneflow.nn as nn
from typing import Optional, Callable, Dict, Union, Any

from .transformer_of import OFEncoder
from .utils_of import OFPreTrainedModel, get_activation

from transformer.builder import OF_MODELS


class OFBertEmbeddings(nn.Module):

    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: flow.Tensor,
        token_type_ids: Optional[flow.Tensor] = None,
        position_ids: Optional[flow.Tensor] = None,
    ) -> flow.Tensor:
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = flow.zeros(
                input_shape, dtype=flow.long, device=input_ids.device)
        if position_ids is None:
            position_ids = flow.arange(
                seq_length, dtype=flow.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + position_embeddings + \
            token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class OFBertPooler(nn.Module):

    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: flow.Tensor) -> flow.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@OF_MODELS.register_module()
class OFBertModel(OFPreTrainedModel):

    def __init__(self, config: Union[Dict[str, Any], Callable[..., None]],
                 **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.embeddings = OFBertEmbeddings(self.config)
        self.encoder = OFEncoder(
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            config.hidden_act,
        )
        self.pooler = OFBertPooler(self.config)

    def forward(
        self,
        inputs: Dict[str, flow.Tensor],
    ) -> Dict[str, flow.Tensor]:
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        token_type_ids = inputs.get("token_type_ids")
        position_ids = inputs.get("position_ids")

        embeddings = self.embeddings(input_ids, token_type_ids, position_ids)

        if attention_mask is None:
            attention_mask = flow.ones_like(input_ids, device=input_ids.device)
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_ids)

        encoder_output, attention_output = self.encoder(
            embeddings, extended_attention_mask)
        pooled_output = self.pooler(encoder_output)
        output_dict = {
            "encoder_output": encoder_output,
            "pooled_output": pooled_output
        }

        return output_dict


class OFBertPredictionHeadTransform(nn.Module):

    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = get_activation(config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=1e-12)

    def forward(self, hidden_states: flow.Tensor) -> flow.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class OFBertLMPredictionHead(nn.Module):

    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.transform = OFBertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden_states: flow.Tensor) -> flow.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class OFBertOnlyMLMHead(nn.Module):

    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.predictions = OFBertLMPredictionHead(config)

    def forward(self, sequence_output: flow.Tensor) -> flow.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@OF_MODELS.register_module()
class OFBertForPreTraining(OFPreTrainedModel):

    def __init__(
        self,
        config: Union[Dict[str, Any], Callable[..., None]],
        model_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self.bert = OFBertModel(self.config, **kwargs)
        self.cls = OFBertOnlyMLMHead(self.config)
        if model_path is not None:
            self._load_weights(model_path)

    def forward(
        self,
        inputs=Dict[str, flow.Tensor],
    ) -> Dict[str, flow.Tensor]:
        output_dict = self.bert(inputs)

        prediction_scores = self.cls(output_dict["encoder_output"])
        output_dict.update({"prediction_scores": prediction_scores})

        return output_dict

    def _load_weights(self, model_path: str = None) -> None:
        _, _ = self.load_state_dict(flow.load(model_path), strict=False)
        self.eval()


def load_tf_weights_in_bert(
    model: "nn.Module",
    config: Callable[..., None],
    tf_checkpoint_path: str,
    with_mlm: bool = True,
) -> None:
    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        raise ImportError(
            "cannot import tensorflow, please install tensorflow first")

    tf_model = tf.train.load_checkpoint(tf_checkpoint_path)

    def _load_tf_variable(key: str) -> np.ndarray:
        return tf_model.get_tensor(key).squeeze()

    def _load_torch_weight(param: flow.nn.Parameter, data: np.ndarray) -> None:
        assert param.shape == data.shape
        param.copy_(nn.Parameter(flow.tensor(data, dtype=flow.float32)))

    def _load_embedding(embedding: "nn.Module", embedding_path: str) -> None:
        embedding_weight = _load_tf_variable(embedding_path)
        _load_torch_weight(embedding.weight, embedding_weight)

    def _load_layer_norm(layer_norm: "nn.Module",
                         layer_norm_base: str) -> None:
        layer_norm_gamma = _load_tf_variable(f"{layer_norm_base}/gamma")
        layer_norm_beta = _load_tf_variable(f"{layer_norm_base}/beta")
        _load_torch_weight(layer_norm.weight, layer_norm_gamma)
        _load_torch_weight(layer_norm.bias, layer_norm_beta)

    def _load_linear(linear: "nn.Module", linear_path: str) -> None:
        linear_weight = _load_tf_variable(f"{linear_path}/kernel")
        linear_weight = np.transpose(linear_weight)
        linear_bias = _load_tf_variable(f"{linear_path}/bias")
        _load_torch_weight(linear.weight, linear_weight)
        _load_torch_weight(linear.bias, linear_bias)

    def _load_self_attention(attention: "nn.Module",
                             attention_path: str) -> None:
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
        model.bert.embeddings.token_embeddings,
        "bert/embeddings/word_embeddings",
    )
    _load_embedding(
        model.bert.embeddings.token_type_embeddings,
        "bert/embeddings/token_type_embeddings",
    )
    _load_embedding(
        model.bert.embeddings.position_embeddings,
        "bert/embeddings/position_embeddings",
    )
    _load_layer_norm(model.bert.embeddings.layer_norm,
                     "bert/embeddings/LayerNorm")

    # loading transformer encoders
    for layer_idx in range(config.num_hidden_layers):
        encoder_layer = model.bert.encoder.layers[layer_idx]
        encoder_path = f"bert/encoder/layer_{layer_idx}"

        _load_self_attention(encoder_layer.self, f"{encoder_path}/attention")
        _load_linear(encoder_layer.self.dense,
                     f"{encoder_path}/attention/output/dense")
        _load_layer_norm(
            encoder_layer.add_norm[0].layer_norm,
            f"{encoder_path}/attention/output/LayerNorm",
        )

        _load_linear(
            encoder_layer.feed_forward.intermediate,
            f"{encoder_path}/intermediate/dense",
        )
        _load_linear(encoder_layer.feed_forward.output,
                     f"{encoder_path}/output/dense")
        _load_layer_norm(
            encoder_layer.add_norm[1].layer_norm,
            f"{encoder_path}/output/LayerNorm",
        )

    _load_linear(model.bert.pooler.dense, "bert/pooler/dense")

    # MLM
    if not with_mlm:
        return

    _load_embedding(model.cls.predictions.decoder,
                    "bert/embeddings/word_embeddings")
    output_bias = _load_tf_variable("cls/predictions/output_bias")
    _load_torch_weight(model.cls.predictions.decoder.bias, output_bias)

    _load_linear(model.cls.predictions.transform.dense,
                 "cls/predictions/transform/dense")
    _load_layer_norm(
        model.cls.predictions.transform.layer_norm,
        "cls/predictions/transform/LayerNorm",
    )


def load_huggingface_weights_in_bert(
    model: "nn.Module",
    config: Callable[..., None],
    pytorch_model_path: str,
    with_mlm: bool = True,
) -> None:
    try:
        from transformers import BertForPreTraining
    except ImportError:
        raise ImportError(
            "cannot import transformers, please install transformers first")

    import numpy as np

    def _load_of_weight(param: flow.nn.Parameter, data: np.ndarray):
        assert param.shape == data.shape
        param.copy_(nn.Parameter(flow.tensor(data, dtype=flow.float32)))

    hf_model = BertForPreTraining.from_pretrained(pytorch_model_path)
    _load_of_weight(
        model.bert.embeddings.token_embeddings.weight,
        hf_model.bert.embeddings.word_embeddings.weight.detach().numpy(),
    )
    _load_of_weight(
        model.bert.embeddings.position_embeddings.weight,
        hf_model.bert.embeddings.position_embeddings.weight.detach().numpy(),
    )
    _load_of_weight(
        model.bert.embeddings.token_type_embeddings.weight,
        hf_model.bert.embeddings.token_type_embeddings.weight.detach().numpy(),
    )
    _load_of_weight(
        model.bert.embeddings.layer_norm.weight,
        hf_model.bert.embeddings.LayerNorm.weight.detach().numpy(),
    )
    _load_of_weight(
        model.bert.embeddings.layer_norm.bias,
        hf_model.bert.embeddings.LayerNorm.bias.detach().numpy(),
    )

    for layer_idx in range(config.num_hidden_layers):
        layer = model.bert.encoder.layers[layer_idx]
        hf_layer = hf_model.bert.encoder.layer[layer_idx]

        _load_of_weight(
            layer.self.query.weight,
            hf_layer.attention.self.query.weight.detach().numpy(),
        )

        _load_of_weight(
            layer.self.query.bias,
            hf_layer.attention.self.query.bias.detach().numpy(),
        )

        _load_of_weight(
            layer.self.key.weight,
            hf_layer.attention.self.key.weight.detach().numpy(),
        )
        _load_of_weight(
            layer.self.key.bias,
            hf_layer.attention.self.key.bias.detach().numpy(),
        )

        _load_of_weight(
            layer.self.value.weight,
            hf_layer.attention.self.value.weight.detach().numpy(),
        )
        _load_of_weight(
            layer.self.value.bias,
            hf_layer.attention.self.value.bias.detach().numpy(),
        )

        _load_of_weight(
            layer.self.dense.weight,
            hf_layer.attention.output.dense.weight.detach().numpy(),
        )
        _load_of_weight(
            layer.self.dense.bias,
            hf_layer.attention.output.dense.bias.detach().numpy(),
        )

        _load_of_weight(
            layer.feed_forward.intermediate.weight,
            hf_layer.intermediate.dense.weight.detach().numpy(),
        )
        _load_of_weight(
            layer.feed_forward.intermediate.bias,
            hf_layer.intermediate.dense.bias.detach().numpy(),
        )

        _load_of_weight(
            layer.feed_forward.output.weight,
            hf_layer.output.dense.weight.detach().numpy(),
        )
        _load_of_weight(
            layer.feed_forward.output.bias,
            hf_layer.output.dense.bias.detach().numpy(),
        )

        _load_of_weight(
            layer.add_norm[0].layer_norm.weight,
            hf_layer.attention.output.LayerNorm.weight.detach().numpy(),
        )
        _load_of_weight(
            layer.add_norm[0].layer_norm.bias,
            hf_layer.attention.output.LayerNorm.bias.detach().numpy(),
        )
        _load_of_weight(
            layer.add_norm[1].layer_norm.weight,
            hf_layer.output.LayerNorm.weight.detach().numpy(),
        )
        _load_of_weight(
            layer.add_norm[1].layer_norm.bias,
            hf_layer.output.LayerNorm.bias.detach().numpy(),
        )

    _load_of_weight(
        model.bert.pooler.dense.weight,
        hf_model.bert.pooler.dense.weight.detach().numpy(),
    )
    _load_of_weight(
        model.bert.pooler.dense.bias,
        hf_model.bert.pooler.dense.bias.detach().numpy(),
    )

    # mlm
    if not with_mlm:
        return

    _load_of_weight(
        model.cls.predictions.decoder.weight,
        hf_model.bert.embeddings.word_embeddings.weight.detach().numpy(),
    )
    _load_of_weight(
        model.cls.predictions.decoder.bias,
        hf_model.cls.predictions.decoder.bias.detach().numpy(),
    )
    _load_of_weight(
        model.cls.predictions.transform.dense.weight,
        hf_model.cls.predictions.transform.dense.weight.detach().numpy(),
    )
    _load_of_weight(
        model.cls.predictions.transform.dense.bias,
        hf_model.cls.predictions.transform.dense.bias.detach().numpy(),
    )
    _load_of_weight(
        model.cls.predictions.transform.layer_norm.weight,
        hf_model.cls.predictions.transform.LayerNorm.weight.detach().numpy(),
    )
    _load_of_weight(
        model.cls.predictions.transform.layer_norm.bias,
        hf_model.cls.predictions.transform.LayerNorm.bias.detach().numpy(),
    )
