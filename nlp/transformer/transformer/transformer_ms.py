import math

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from typing import Tuple, Optional

from .utils_ms import get_activation


class Encoder(nn.Cell):
    def __init__(
        self,
        num_hidden_layers: int,
        d_model: int,
        nhead: int,
        d_feedforward: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        hidden_act: str = "relu",
    ) -> None:
        super().__init__()
        self.layers = nn.CellList(
            [
                EncoderLayer(
                    d_model,
                    nhead,
                    d_feedforward,
                    attention_probs_dropout_prob,
                    hidden_dropout_prob,
                    hidden_act,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def construct(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        attn = None
        for layer in self.layers:
            x, attn = layer(x, mask)
        return x, attn


class EncoderLayer(nn.Cell):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_feedforward: int = 2048,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, attention_probs_dropout_prob)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_feedforward, hidden_dropout_prob, hidden_act
        )
        self.add_norm = nn.CellList([AddNormLayer(d_model, hidden_dropout_prob) for _ in range(2)])

    def construct(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, ...]:
        x1, attn = self.self_attn(x, x, x, mask)
        x = self.add_norm[0](x, x1)
        x1 = self.feed_forward(x)
        x = self.add_norm[1](x, x1)
        return x, attn


class AddNormLayer(nn.Cell):
    def __init__(self, d_model: int, hidden_dropout_prob: int = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm((d_model,), epsilon=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def construct(self, x: Tensor, x1: Tensor) -> Tensor:
        return self.layer_norm(x + self.dropout(x1))


class PositionwiseFeedForward(nn.Cell):
    def __init__(
        self,
        d_model: int,
        d_feedforward: int,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "relu",
    ) -> None:
        super().__init__()
        self.hidden_act = hidden_act
        self.intermediate = nn.Dense(d_model, d_feedforward)
        self.output = nn.Dense(d_feedforward, d_model)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def construct(self, x: Tensor) -> Tensor:
        return self.output(self.dropout(get_activation(self.hidden_act)(self.intermediate(x))))


class MultiHeadAttention(nn.Cell):
    def __init__(
        self, d_model: int = 512, n_head: int = 8, attention_probs_dropout_prob: float = 0.1
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.n_head = n_head
        self.query = nn.Dense(d_model, d_model)
        self.key = nn.Dense(d_model, d_model)
        self.value = nn.Dense(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_per_head, attention_probs_dropout_prob)
        self.dense = nn.Dense(d_model, d_model)

    def construct(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = query.shape[0]

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # multi head
        query = query.view(batch_size, -1, self.n_head, self.d_per_head).swapaxes(1, 2)
        key = key.view(batch_size, -1, self.n_head, self.d_per_head).transpose(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.n_head, self.d_per_head).transpose(0, 2, 1, 3)

        # self attention
        context, attention = self.attention(query, key, value, attn_mask=mask)
        # concat heads
        context = context.transpose(0, 2, 1, 3).view(batch_size, -1, self.d_model)
        output = self.dense(context)

        return output, attention


class ScaledDotProductAttention(nn.Cell):
    def __init__(self, factor: int, attention_probs_dropout_prob: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.factor = math.sqrt(factor)

    def construct(
        self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            query: [batch, len_query, dim_query]
            key: [batch, len_key, dim_key]
            value: [batch, len_value, dim_value]
            attn_mask: [batch, len_query, len_key]
        """
        attention = ops.matmul(query, key.swapaxes(-1, -2))
        attention = attention / self.factor
        if attn_mask is not None:
            attention = attention + attn_mask
        attention = ops.Softmax(axis=-1)(attention)
        attention = self.dropout(attention)
        context = ops.matmul(attention, value)
        return context, attention
