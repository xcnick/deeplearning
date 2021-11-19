import math

import paddle
import paddle.nn as nn
from typing import Tuple, Optional

from .utils_pd import get_activation


class PDEncoder(nn.Layer):

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
        self.layers = nn.LayerList([
            EncoderLayer(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                attention_probs_dropout_prob,
                hidden_dropout_prob,
                hidden_act,
            ) for _ in range(num_hidden_layers)
        ])

    def forward(
        self,
        x: paddle.Tensor,
        mask: Optional[paddle.Tensor] = None
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        for layer in self.layers:
            x, attn = layer(x, mask)
        return x, attn


class EncoderLayer(nn.Layer):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int = 3072,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "relu",
    ) -> None:
        super().__init__()
        self.self = MultiHeadAttention(hidden_size, num_attention_heads,
                                       attention_probs_dropout_prob)
        self.feed_forward = PositionwiseFeedForward(hidden_size,
                                                    intermediate_size,
                                                    hidden_dropout_prob,
                                                    hidden_act)
        self.add_norm = nn.LayerList(
            [AddNormLayer(hidden_size, hidden_dropout_prob) for _ in range(2)])

    def forward(
        self,
        x: paddle.Tensor,
        mask: Optional[paddle.Tensor] = None
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        x1, attn = self.self(x, x, x, mask)
        x = self.add_norm[0](x, x1)
        x1 = self.feed_forward(x)
        x = self.add_norm[1](x, x1)
        return x, attn


class AddNormLayer(nn.Layer):

    def __init__(self,
                 hidden_size: int,
                 hidden_dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: paddle.Tensor, x1: paddle.Tensor) -> paddle.Tensor:
        return self.layer_norm(x + self.dropout(x1))


class PositionwiseFeedForward(nn.Layer):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "relu",
    ) -> None:
        super().__init__()
        self.hidden_act = hidden_act
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.output(
            self.dropout(
                get_activation(self.hidden_act)(self.intermediate(x))))


class MultiHeadAttention(nn.Layer):

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
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.attention = ScaledDotProductAttention(
            attention_probs_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        mask: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        batch_size = query.shape[0]

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # multi head
        query = query.reshape((batch_size, -1, self.num_attention_heads,
                               self.dims_per_head)).transpose((0, 2, 1, 3))
        key = key.reshape((batch_size, -1, self.num_attention_heads,
                           self.dims_per_head)).transpose((0, 2, 1, 3))
        value = value.reshape((batch_size, -1, self.num_attention_heads,
                               self.dims_per_head)).transpose((0, 2, 1, 3))

        # self attention
        context, attention = self.attention(query, key, value, attn_mask=mask)
        # concat heads
        context = context.transpose((0, 2, 1, 3)).reshape(
            (batch_size, -1, self.hidden_size))
        output = self.dense(context)

        return output, attention


class ScaledDotProductAttention(nn.Layer):

    def __init__(self, attention_probs_dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def forward(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        attn_mask: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        r"""
        Args:
            query: [batch, num_attention_heads, len_query, dim_query]
            key: [batch, num_attention_heads, len_key, dim_key]
            value: [batch, num_attention_heads, len_value, dim_value]
            attn_mask: [batch, num_attention_heads, len_query, len_key]
        """
        attention = paddle.matmul(query, key.transpose((0, 1, 3, 2)))
        attention = attention / math.sqrt(query.shape[-1])
        if attn_mask is not None:
            attention = attention + attn_mask
        attention = nn.Softmax(axis=-1)(attention)
        attention = self.dropout(attention)
        context = paddle.matmul(attention, value)
        return context, attention
