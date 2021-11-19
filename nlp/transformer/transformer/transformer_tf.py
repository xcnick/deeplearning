import tensorflow as tf

from typing import Tuple, Optional
from .utils_tf import get_activation


class TFEncoder(tf.keras.layers.Layer):

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        hidden_act: str = "relu",
    ) -> None:
        super().__init__()
        self.layers = [
            TFEncoderLayer(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                attention_probs_dropout_prob,
                hidden_dropout_prob,
                hidden_act,
                name="layer_{}".format(i),
            ) for i in range(num_hidden_layers)
        ]

    def call(self,
             x: tf.Tensor,
             mask: Optional[tf.Tensor],
             training: Optional[bool] = False) -> Tuple[tf.Tensor, ...]:
        for layer in self.layers:
            x, attn = layer(x, mask, training=training)
        return x, attn


class TFEncoderLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int = 3072,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.self = TFMultiHeadAttention(hidden_size, num_attention_heads,
                                         attention_probs_dropout_prob)
        self.feed_forward = TFPositionwiseFeedForward(hidden_size,
                                                      intermediate_size,
                                                      hidden_dropout_prob,
                                                      hidden_act)
        self.add_norm = [
            TFAddNormLayer(hidden_dropout_prob, name="add_norm_{}".format(i))
            for i in range(2)
        ]

    def call(self,
             x: tf.Tensor,
             mask: Optional[tf.Tensor] = None,
             training: Optional[bool] = False) -> Tuple[tf.Tensor, ...]:
        x1, attn = self.self(x, x, x, mask, training=training)
        x = self.add_norm[0](x, x1, training=training)
        x1 = self.feed_forward(x, training=training)
        x = self.add_norm[1](x, x1, training=training)
        return x, attn


class TFAddNormLayer(tf.keras.layers.Layer):

    def __init__(self, hidden_dropout_prob: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-12, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_prob)

    def call(self,
             x: tf.Tensor,
             x1: tf.Tensor,
             training: Optional[bool] = False):
        return self.layer_norm(x + self.dropout(x1, training=training))


class TFPositionwiseFeedForward(tf.keras.layers.Layer):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout_prob: Optional[float] = 0.1,
        hidden_act: Optional[str] = "relu",
    ) -> None:
        super().__init__()
        self.hidden_act = hidden_act
        self.intermediate = tf.keras.layers.Dense(
            intermediate_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name="intermediate",
        )
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name="dense",
        )
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_prob)

    def call(self,
             x: tf.Tensor,
             training: Optional[bool] = False) -> tf.Tensor:
        return self.dense(
            self.dropout(
                get_activation(self.hidden_act)(self.intermediate(x)),
                training=training))


class TFMultiHeadAttention(tf.keras.layers.Layer):

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.dims_per_head = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.query = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name="value",
        )

        self.attention = TFScaledDotProductAttention(
            attention_probs_dropout_prob)
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            name="dense",
        )

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = False,
    ) -> Tuple[tf.Tensor, ...]:
        batch_size = tf.shape(query)[0]

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # multi head
        query = tf.reshape(
            query,
            (batch_size, -1, self.num_attention_heads, self.dims_per_head))
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.reshape(
            key,
            (batch_size, -1, self.num_attention_heads, self.dims_per_head))
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.reshape(
            value,
            (batch_size, -1, self.num_attention_heads, self.dims_per_head))
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        # self attention
        context, attention = self.attention(
            query, key, value, attn_mask=mask, training=training)
        # concat heads
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.hidden_size))
        output = self.dense(context)

        return output, attention


class TFScaledDotProductAttention(tf.keras.layers.Layer):

    def __init__(self, attention_probs_dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(attention_probs_dropout_prob)

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        attn_mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = False,
    ) -> Tuple[tf.Tensor, ...]:
        r"""
        Args:
            query: [batch, num_attention_heads, len_query, dim_query]
            key: [batch, num_attention_heads, len_key, dim_key]
            value: [batch, num_attention_heads, len_value, dim_value]
            attn_mask: [batch, num_attention_heads, len_query, len_key]
        """
        attention = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(query.shape[-1], attention.dtype)
        attention = attention / tf.math.sqrt(dk)  # 缩放因子为 \sqrt{dims_per_head}
        if attn_mask is not None:
            attention = attention + attn_mask
        attention = tf.nn.softmax(attention, axis=-1)
        attention = self.dropout(attention, training=training)
        context = tf.matmul(attention, value)
        return context, attention
