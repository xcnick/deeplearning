import paddle
import paddle.nn as nn
from typing import Optional, Callable, Dict, Union, Any

from .transformer_pd import PDEncoder
from .utils_pd import PDPreTrainedModel, get_activation

from transformer.builder import PD_MODELS


class PDBertEmbeddings(nn.Layer):
    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: paddle.Tensor,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        input_shape = input_ids.shape
        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.long, device=input_ids.device)
        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype=paddle.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class PDBertPooler(nn.Layer):
    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@PD_MODELS.register_module()
class PDBertModel(PDPreTrainedModel):
    def __init__(self, config: Union[Dict[str, Any], Callable[..., None]], **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.embeddings = PDBertEmbeddings(self.config)
        self.encoder = PDEncoder(
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            config.hidden_act,
        )
        self.pooler = PDBertPooler(self.config)

    def forward(self, inputs: Dict[str, paddle.Tensor],) -> Dict[str, paddle.Tensor]:
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        token_type_ids = inputs.get("token_type_ids")
        position_ids = inputs.get("position_ids")

        embeddings = self.embeddings(input_ids, token_type_ids, position_ids)

        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids, device=input_ids.device)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids)

        encoder_output, attention_output = self.encoder(embeddings, extended_attention_mask)
        pooled_output = self.pooler(encoder_output)
        output_dict = {"encoder_output": encoder_output, "pooled_output": pooled_output}

        return output_dict


class PDBertPredictionHeadTransform(nn.Layer):
    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = get_activation(config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=1e-12)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class PDBertLMPredictionHead(nn.Layer):
    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.transform = PDBertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class PDBertOnlyMLMHead(nn.Layer):
    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.predictions = PDBertLMPredictionHead(config)

    def forward(self, sequence_output: paddle.Tensor) -> paddle.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@PD_MODELS.register_module()
class PDBertForPreTraining(PDPreTrainedModel):
    def __init__(
        self,
        config: Union[Dict[str, Any], Callable[..., None]],
        model_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self.bert = PDBertModel(self.config, **kwargs)
        self.cls = PDBertOnlyMLMHead(self.config)
        if model_path is not None:
            self._load_weights(model_path)

    def forward(self, inputs=Dict[str, paddle.Tensor],) -> Dict[str, paddle.Tensor]:
        output_dict = self.bert(inputs)

        prediction_scores = self.cls(output_dict["encoder_output"])
        output_dict.update({"prediction_scores": prediction_scores})

        return output_dict

    def _load_weights(self, model_path: str = None) -> None:
        self.set_state_dict(paddle.load(model_path))
        self.eval()


def load_tf_weights_in_bert(
    model: "nn.Layer", config: Callable[..., None], tf_checkpoint_path: str, with_mlm: bool = True,
) -> None:
    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        raise ImportError("cannot import tensorflow, please install tensorflow first")

    tf_model = tf.train.load_checkpoint(tf_checkpoint_path)

    def _load_tf_variable(key: str) -> np.ndarray:
        return tf_model.get_tensor(key).squeeze()

    def _load_paddle_weight(param: paddle.Tensor, data: np.ndarray) -> None:
        assert param.shape == list(data.shape)
        param.set_value(data.astype(np.float32))

    def _load_embedding(embedding: "nn.Embedding", embedding_path: str) -> None:
        embedding_weight = _load_tf_variable(embedding_path)
        _load_paddle_weight(embedding.weight, embedding_weight)

    def _load_layer_norm(layer_norm: "nn.LayerNorm", layer_norm_base: str) -> None:
        layer_norm_gamma = _load_tf_variable(f"{layer_norm_base}/gamma")
        layer_norm_beta = _load_tf_variable(f"{layer_norm_base}/beta")
        _load_paddle_weight(layer_norm.weight, layer_norm_gamma)
        _load_paddle_weight(layer_norm.bias, layer_norm_beta)

    def _load_linear(linear: "nn.Linear", linear_path: str) -> None:
        linear_weight = _load_tf_variable(f"{linear_path}/kernel")
        linear_bias = _load_tf_variable(f"{linear_path}/bias")
        _load_paddle_weight(linear.weight, linear_weight)
        _load_paddle_weight(linear.bias, linear_bias)

    def _load_self_attention(attention: "nn.Layer", attention_path: str) -> None:
        query_weight = _load_tf_variable(f"{attention_path}/self/query/kernel")
        key_weight = _load_tf_variable(f"{attention_path}/self/key/kernel")
        value_weight = _load_tf_variable(f"{attention_path}/self/value/kernel")

        query_bias = _load_tf_variable(f"{attention_path}/self/query/bias")
        key_bias = _load_tf_variable(f"{attention_path}/self/key/bias")
        value_bias = _load_tf_variable(f"{attention_path}/self/value/bias")

        _load_paddle_weight(attention.query.weight, query_weight)
        _load_paddle_weight(attention.key.weight, key_weight)
        _load_paddle_weight(attention.value.weight, value_weight)

        _load_paddle_weight(attention.query.bias, query_bias)
        _load_paddle_weight(attention.key.bias, key_bias)
        _load_paddle_weight(attention.value.bias, value_bias)

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

    _load_linear(model.bert.pooler.dense, "bert/pooler/dense")

    # MLM
    if not with_mlm:
        return

    decoder_weight = _load_tf_variable("bert/embeddings/word_embeddings")
    _load_paddle_weight(model.cls.predictions.decoder.weight, np.transpose(decoder_weight))

    output_bias = _load_tf_variable("cls/predictions/output_bias")
    _load_paddle_weight(model.cls.predictions.decoder.bias, output_bias)

    _load_linear(model.cls.predictions.transform.dense, "cls/predictions/transform/dense")
    _load_layer_norm(
        model.cls.predictions.transform.layer_norm, "cls/predictions/transform/LayerNorm",
    )


def load_huggingface_weights_in_bert(
    model: "nn.Layer", config: Callable[..., None], torch_model_path: str, with_mlm: bool = True,
) -> None:
    try:
        from transformers import BertForPreTraining
    except ImportError:
        raise ImportError("cannot import transformers, please install transformers first")

    import numpy as np

    def _load_paddle_weight(param: paddle.Tensor, data: np.ndarray):
        assert param.shape == list(data.shape)
        param.set_value(data.astype(np.float32))

    def _load_embedding(embedding: "nn.Embedding", torch_embedding: "torch.nn.Embedding") -> None:
        _load_paddle_weight(embedding.weight, torch_embedding.weight.detach().numpy())

    def _load_layer_norm(
        layer_norm: "nn.LayerNorm", torch_layer_norm: "torch.nn.LayerNorm"
    ) -> None:
        _load_paddle_weight(layer_norm.weight, torch_layer_norm.weight.detach().numpy())
        _load_paddle_weight(layer_norm.bias, torch_layer_norm.bias.detach().numpy())

    def _load_linear(linear: "nn.Linear", torch_linear: "torch.nn.Linear") -> None:
        _load_paddle_weight(linear.weight, torch_linear.weight.transpose(-1, 0).detach().numpy())
        _load_paddle_weight(linear.bias, torch_linear.bias.transpose(-1, 0).detach().numpy())

    def _load_self_attention(attention: "nn.Linear", torch_attention: "torch.nn.Module") -> None:
        _load_linear(attention.query, torch_attention.self.query)
        _load_linear(attention.key, torch_attention.self.key)
        _load_linear(attention.value, torch_attention.self.value)

    hf_model = BertForPreTraining.from_pretrained(torch_model_path)
    _load_embedding(
        model.bert.embeddings.token_embeddings, hf_model.bert.embeddings.word_embeddings,
    )
    _load_embedding(
        model.bert.embeddings.position_embeddings, hf_model.bert.embeddings.position_embeddings,
    )
    _load_embedding(
        model.bert.embeddings.token_type_embeddings, hf_model.bert.embeddings.token_type_embeddings,
    )

    _load_layer_norm(model.bert.embeddings.layer_norm, hf_model.bert.embeddings.LayerNorm)

    for layer_idx in range(config.num_hidden_layers):
        layer = model.bert.encoder.layers[layer_idx]
        hf_layer = hf_model.bert.encoder.layer[layer_idx]

        _load_self_attention(layer.self, hf_layer.attention)

        _load_linear(layer.self.dense, hf_layer.attention.output.dense)
        _load_linear(layer.feed_forward.intermediate, hf_layer.intermediate.dense)
        _load_linear(layer.feed_forward.output, hf_layer.output.dense)

        _load_layer_norm(layer.add_norm[0].layer_norm, hf_layer.attention.output.LayerNorm)
        _load_layer_norm(layer.add_norm[1].layer_norm, hf_layer.output.LayerNorm)

    _load_linear(model.bert.pooler.dense, hf_model.bert.pooler.dense)

    # mlm
    if not with_mlm:
        return

    _load_paddle_weight(
        model.cls.predictions.decoder.weight,
        hf_model.bert.embeddings.word_embeddings.weight.transpose(-1, 0).detach().numpy(),
    )
    _load_paddle_weight(
        model.cls.predictions.decoder.bias, hf_model.cls.predictions.decoder.bias.detach().numpy()
    )

    _load_linear(model.cls.predictions.transform.dense, hf_model.cls.predictions.transform.dense)

    _load_layer_norm(
        model.cls.predictions.transform.layer_norm, hf_model.cls.predictions.transform.LayerNorm
    )
