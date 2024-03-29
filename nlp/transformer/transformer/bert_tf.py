import tensorflow as tf
from typing import Optional, Callable, Dict, Union, Any

from .transformer_tf import TFEncoder
from .utils_tf import get_activation, TFPreTrainedModel

from transformer.builder import TF_MODELS


class TFBertEmbeddings(tf.keras.layers.Layer):

    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name="token_embeddings",
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name="position_embeddings",
        )
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name="token_type_embeddings",
        )

        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-12, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(
        self,
        input_ids: tf.Tensor,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        input_shape = tf.shape(input_ids)
        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=tf.int32)
        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]

        input_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + position_embeddings + \
            token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class TFBertPooler(tf.keras.layers.Layer):

    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            activation="tanh",
            name="dense",
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


@TF_MODELS.register_module()
class TFBertModel(TFPreTrainedModel):

    def __init__(self, config: Union[Dict[str, Any], Callable[..., None]],
                 **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.embeddings = TFBertEmbeddings(self.config)
        self.encoder = TFEncoder(
            self.config.num_hidden_layers,
            self.config.hidden_size,
            self.config.num_attention_heads,
            self.config.intermediate_size,
            self.config.attention_probs_dropout_prob,
            self.config.hidden_dropout_prob,
            self.config.hidden_act,
        )
        self.pooler = TFBertPooler(self.config)

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: Optional[bool] = False,
    ) -> Dict[str, tf.Tensor]:
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        token_type_ids = inputs.get("token_type_ids")
        position_ids = inputs.get("position_ids")

        if attention_mask is None:
            attention_mask = tf.ones_like(input_ids)

        embeddings = self.embeddings(
            input_ids, token_type_ids, position_ids, training=training)

        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        extended_attention_mask = tf.cast(
            extended_attention_mask, dtype=embeddings.dtype)
        one_cst = tf.constant(1.0, dtype=embeddings.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embeddings.dtype)
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = tf.multiply(
            tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        encoder_output, attention_output = self.encoder(
            embeddings, extended_attention_mask, training=training)
        pooled_output = self.pooler(encoder_output)
        output_dict = {
            "encoder_output": encoder_output,
            "pooled_output": pooled_output
        }

        return output_dict


class TFBertPredictionHeadTransform(tf.keras.layers.Layer):

    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name="dense",
        )
        self.transform_act_fn = get_activation(config.hidden_act)
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-12, name="layer_norm")

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class TFBertLMPredictionHead(tf.keras.layers.Layer):

    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.transform = TFBertPredictionHeadTransform(config)
        self.decoder = tf.keras.layers.Dense(
            config.vocab_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name="dense",
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class TFBertOnlyMLMHead(tf.keras.layers.Layer):

    def __init__(self, config: Callable[..., None]) -> None:
        super().__init__()
        self.predictions = TFBertLMPredictionHead(config)

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@TF_MODELS.register_module()
class TFBertForPreTraining(TFPreTrainedModel):

    def __init__(
        self,
        config: Union[Dict[str, Any], Callable[..., None]],
        model_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self.bert = TFBertModel(self.config, **kwargs)
        self.cls = TFBertOnlyMLMHead(self.config)
        if model_path is not None:
            self._load_weights(model_path)

    def call(self,
             inputs=Dict[str, tf.Tensor],
             training: Optional[bool] = False) -> Dict[str, tf.Tensor]:
        output_dict = self.bert(inputs, training=training)

        prediction_scores = self.cls(output_dict["encoder_output"])
        output_dict.update({"prediction_scores": prediction_scores})

        return output_dict

    def _load_weights(self, model_path: str = None) -> None:
        inputs = {
            "input_ids": tf.zeros((1, 1), dtype=tf.int32),
        }
        self(inputs, training=False)
        self.load_weights(model_path)


def load_tf_weights_in_bert_to_tf(
    model: "tf.keras.Model",
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

    def _load_tf_variable(key):
        return tf_model.get_tensor(key).squeeze()

    def _set_weights(param, weight_list):
        assert len(param.weights) == len(weight_list)
        for param_weight, weight in zip(param.weights, weight_list):
            assert param_weight.shape == weight.shape
        param.set_weights(weight_list)

    def _load_embedding(embedding, embedding_path):
        embedding_weight = _load_tf_variable(embedding_path)
        _set_weights(embedding, [embedding_weight])

    def _load_layer_norm(layer_norm, layer_norm_base):
        layer_norm_gamma = _load_tf_variable(f"{layer_norm_base}/gamma")
        layer_norm_beta = _load_tf_variable(f"{layer_norm_base}/beta")
        _set_weights(layer_norm, [layer_norm_gamma, layer_norm_beta])

    def _load_linear(linear, linear_path):
        linear_weight = _load_tf_variable(f"{linear_path}/kernel")
        linear_bias = _load_tf_variable(f"{linear_path}/bias")
        _set_weights(linear, [linear_weight, linear_bias])

    def _load_self_attention(attention, attention_path):
        query_weight = _load_tf_variable(f"{attention_path}/self/query/kernel")
        key_weight = _load_tf_variable(f"{attention_path}/self/key/kernel")
        value_weight = _load_tf_variable(f"{attention_path}/self/value/kernel")

        query_bias = _load_tf_variable(f"{attention_path}/self/query/bias")
        key_bias = _load_tf_variable(f"{attention_path}/self/key/bias")
        value_bias = _load_tf_variable(f"{attention_path}/self/value/bias")

        _set_weights(attention.query, [query_weight, query_bias])
        _set_weights(attention.key, [key_weight, key_bias])
        _set_weights(attention.value, [value_weight, value_bias])

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
        _load_linear(encoder_layer.feed_forward.dense,
                     f"{encoder_path}/output/dense")
        _load_layer_norm(
            encoder_layer.add_norm[1].layer_norm,
            f"{encoder_path}/output/LayerNorm",
        )

    _load_linear(model.bert.pooler.dense, "bert/pooler/dense")

    # MLM
    if not with_mlm:
        return

    output_embeddings = _load_tf_variable("bert/embeddings/word_embeddings")
    output_embeddings = np.transpose(output_embeddings)
    output_bias = _load_tf_variable("cls/predictions/output_bias")
    _set_weights(model.cls.predictions.decoder,
                 [output_embeddings, output_bias])

    _load_linear(model.cls.predictions.transform.dense,
                 "cls/predictions/transform/dense")
    _load_layer_norm(
        model.cls.predictions.transform.layer_norm,
        "cls/predictions/transform/LayerNorm",
    )


def load_huggingface_weights_in_bert_to_tf(
    model: "tf.keras.Model",
    config: Callable[..., None],
    pytorch_model_path: str,
    with_mlm: bool = True,
) -> None:
    try:
        from transformers import BertForPreTraining
    except ImportError:
        raise ImportError(
            "cannot import transformers, please install transformers first")

    def _set_weights(param, weight_list):
        assert len(param.weights) == len(weight_list)
        for param_weight, weight in zip(param.weights, weight_list):
            assert param_weight.shape == weight.shape
        param.set_weights(weight_list)

    hf_model = BertForPreTraining.from_pretrained(pytorch_model_path)
    _set_weights(
        model.bert.embeddings.token_embeddings,
        [hf_model.bert.embeddings.word_embeddings.weight.detach().numpy()],
    )
    _set_weights(
        model.bert.embeddings.position_embeddings,
        [hf_model.bert.embeddings.position_embeddings.weight.detach().numpy()],
    )
    _set_weights(
        model.bert.embeddings.token_type_embeddings,
        [
            hf_model.bert.embeddings.token_type_embeddings.weight.detach().
            numpy()
        ],
    )

    _set_weights(
        model.bert.embeddings.layer_norm,
        [
            hf_model.bert.embeddings.LayerNorm.weight.detach().numpy(),
            hf_model.bert.embeddings.LayerNorm.bias.detach().numpy(),
        ],
    )

    for layer_idx in range(config.num_hidden_layers):
        layer = model.bert.encoder.layers[layer_idx]
        hf_layer = hf_model.bert.encoder.layer[layer_idx]

        _set_weights(
            layer.self.query,
            [
                hf_layer.attention.self.query.weight.detach().transpose(
                    0, 1).numpy(),
                hf_layer.attention.self.query.bias.detach().numpy(),
            ],
        )

        _set_weights(
            layer.self.key,
            [
                hf_layer.attention.self.key.weight.detach().transpose(
                    0, 1).numpy(),
                hf_layer.attention.self.key.bias.detach().numpy(),
            ],
        )

        _set_weights(
            layer.self.value,
            [
                hf_layer.attention.self.value.weight.detach().transpose(
                    0, 1).numpy(),
                hf_layer.attention.self.value.bias.detach().numpy(),
            ],
        )

        _set_weights(
            layer.self.dense,
            [
                hf_layer.attention.output.dense.weight.detach().transpose(
                    0, 1).numpy(),
                hf_layer.attention.output.dense.bias.detach().numpy(),
            ],
        )

        _set_weights(
            layer.feed_forward.intermediate,
            [
                hf_layer.intermediate.dense.weight.detach().transpose(
                    0, 1).numpy(),
                hf_layer.intermediate.dense.bias.detach().numpy(),
            ],
        )

        _set_weights(
            layer.feed_forward.dense,
            [
                hf_layer.output.dense.weight.detach().transpose(0, 1).numpy(),
                hf_layer.output.dense.bias.detach().numpy(),
            ],
        )

        _set_weights(
            layer.add_norm[0].layer_norm,
            [
                hf_layer.attention.output.LayerNorm.weight.detach().numpy(),
                hf_layer.attention.output.LayerNorm.bias.detach().numpy(),
            ],
        )

        _set_weights(
            layer.add_norm[1].layer_norm,
            [
                hf_layer.output.LayerNorm.weight.detach().numpy(),
                hf_layer.output.LayerNorm.bias.detach().numpy(),
            ],
        )

    _set_weights(
        model.bert.pooler.dense,
        [
            hf_model.bert.pooler.dense.weight.detach().transpose(0, 1).numpy(),
            hf_model.bert.pooler.dense.bias.detach().numpy(),
        ],
    )

    # mlm
    if not with_mlm:
        return

    _set_weights(
        model.cls.predictions.decoder,
        [
            hf_model.bert.embeddings.word_embeddings.weight.detach().transpose(
                0, 1).numpy(),
            hf_model.cls.predictions.decoder.bias.detach().numpy(),
        ],
    )

    _set_weights(
        model.cls.predictions.transform.dense,
        [
            hf_model.cls.predictions.transform.dense.weight.detach().transpose(
                0, 1).numpy(),
            hf_model.cls.predictions.transform.dense.bias.detach().numpy(),
        ],
    )

    _set_weights(
        model.cls.predictions.transform.layer_norm,
        [
            hf_model.cls.predictions.transform.LayerNorm.weight.detach().numpy(
            ),
            hf_model.cls.predictions.transform.LayerNorm.bias.detach().numpy(),
        ],
    )
