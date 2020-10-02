import copy

import tensorflow as tf

from .utils_tf import get_activation


class TFEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_hidden_layers,
        d_model,
        nhead,
        d_feedforward,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        hidden_act="relu",
    ):
        super().__init__()
        self.layers = [
            TFEncoderLayer(
                d_model,
                nhead,
                d_feedforward,
                attention_probs_dropout_prob,
                hidden_dropout_prob,
                hidden_act,
                name="layer_{}".format(i),
            )
            for i in range(num_hidden_layers)
        ]

    def call(self, x, mask, training=False):
        for layer in self.layers:
            x, attn = layer(x, mask, training=training)
        return x, attn


class TFEncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        nhead,
        d_feedforward=2048,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        hidden_act="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self = TFMultiHeadAttention(d_model, nhead, attention_probs_dropout_prob)
        self.feed_forward = TFPositionwiseFeedForward(
            d_model, d_feedforward, hidden_dropout_prob, hidden_act
        )
        self.add_norm = [
            TFAddNormLayer(d_model, hidden_dropout_prob, name="add_norm_{}".format(i))
            for i in range(2)
        ]

    def call(self, x, mask=None, training=False):
        x1, attn = self.self(x, x, x, mask, training=training)
        x = self.add_norm[0](x, x1, training=training)
        x1 = self.feed_forward(x, training=training)
        x = self.add_norm[1](x, x1, training=training)
        return x, attn


class TFAddNormLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, hidden_dropout_prob, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_prob)

    def call(self, x, x1, training=False):
        return self.layer_norm(x + self.dropout(x1, training=training))


class TFPositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_feedforward, hidden_dropout_prob=0.1, hidden_act="relu"):
        super().__init__()
        self.hidden_act = hidden_act
        self.intermediate = tf.keras.layers.Dense(
            d_feedforward,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="intermediate",
        )
        self.dense = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="dense",
        )
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_prob)

    def call(self, x, training=False):
        return self.dense(
            self.dropout(get_activation(self.hidden_act)(self.intermediate(x)), training=training)
        )


class TFMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model=512, n_head=8, attention_probs_dropout_prob=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.n_head = n_head
        self.query = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="value",
        )

        self.attention = TFScaledDotProductAttention(attention_probs_dropout_prob)
        self.dense = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="dense",
        )

    def call(self, query, key, value, mask=None, training=False):
        batch_size = query.shape[0]

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # multi head
        query = tf.reshape(query, (batch_size, -1, self.n_head, self.d_per_head))
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.reshape(key, (batch_size, -1, self.n_head, self.d_per_head))
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.reshape(value, (batch_size, -1, self.n_head, self.d_per_head))
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        # self attention
        context, attention = self.attention(query, key, value, attn_mask=mask, training=training)
        # concat heads
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.d_model))
        output = self.dense(context)

        return output, attention


class TFScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, attention_probs_dropout_prob=0.0):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(attention_probs_dropout_prob)

    def call(self, query, key, value, attn_mask=None, training=False):
        r"""
        Args:
            query: [batch, len_query, dim_query]
            key: [batch, len_key, dim_key]
            value: [batch, len_value, dim_value]
            attn_mask: [batch, len_query, len_key]
        """
        attention = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(query.shape[-1], tf.float32)
        attention = attention / tf.math.sqrt(dk)  # 缩放因子为 \sqrt{d_per_head}
        if attn_mask is not None:
            attention = attention + attn_mask
        attention = tf.nn.softmax(attention, axis=-1)
        attention = self.dropout(attention, training=training)
        context = tf.matmul(attention, value)
        return context, attention
