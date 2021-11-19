import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import TruncatedNormal

from typing import Tuple, Optional

from mindspore.ops.primitive import constexpr

from .utils_ms import get_activation


class Encoder(nn.Cell):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        initializer_range: float,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        hidden_act: str = "relu",
    ) -> None:
        super().__init__()
        self.layers = nn.CellList(
            [
                EncoderLayer(
                    hidden_size,
                    num_attention_heads,
                    intermediate_size,
                    initializer_range,
                    attention_probs_dropout_prob,
                    hidden_dropout_prob,
                    hidden_act,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def construct(
        self, x: ms.Tensor, mask: Optional[ms.Tensor] = None
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        attn = None
        for layer in self.layers:
            x, attn = layer(x, mask)
        return x, attn


class EncoderLayer(nn.Cell):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int = 2048,
        initializer_range: float = 0.02,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            hidden_size, num_attention_heads, initializer_range, attention_probs_dropout_prob
        )
        self.feed_forward = PositionwiseFeedForward(
            hidden_size, intermediate_size, initializer_range, hidden_dropout_prob, hidden_act
        )
        self.add_norm = nn.CellList(
            [AddNormLayer(hidden_size, hidden_dropout_prob) for _ in range(2)]
        )

    def construct(self, x: ms.Tensor, mask: Optional[ms.Tensor] = None) -> Tuple[ms.Tensor, ...]:
        x1, attn = self.self_attn(x, x, x, mask)
        x = self.add_norm[0](x, x1)
        x1 = self.feed_forward(x)
        x = self.add_norm[1](x, x1)
        return x, attn


class AddNormLayer(nn.Cell):
    def __init__(self, hidden_size: int, hidden_dropout_prob: int = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm((hidden_size,), epsilon=1e-12)
        self.dropout = nn.Dropout(1.0 - hidden_dropout_prob)

    def construct(self, x: ms.Tensor, x1: ms.Tensor) -> ms.Tensor:
        return self.layer_norm(x + self.dropout(x1))


class PositionwiseFeedForward(nn.Cell):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        initializer_range: float = 0.02,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "relu",
    ) -> None:
        super().__init__()
        self.hidden_act = hidden_act
        self.intermediate = nn.Dense(
            hidden_size, intermediate_size, weight_init=TruncatedNormal(initializer_range)
        )
        self.output = nn.Dense(
            intermediate_size, hidden_size, weight_init=TruncatedNormal(initializer_range)
        )
        self.dropout = nn.Dropout(1.0 - hidden_dropout_prob)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return self.output(self.dropout(get_activation(self.hidden_act)(self.intermediate(x))))


class MultiHeadAttention(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        initializer_range: float = 0.02,
        attention_probs_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.dims_per_head = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.query = nn.Dense(
            hidden_size, hidden_size, weight_init=TruncatedNormal(initializer_range)
        )
        self.key = nn.Dense(
            hidden_size, hidden_size, weight_init=TruncatedNormal(initializer_range)
        )
        self.value = nn.Dense(
            hidden_size, hidden_size, weight_init=TruncatedNormal(initializer_range)
        )

        self.attention = ScaledDotProductAttention(attention_probs_dropout_prob)
        self.dense = nn.Dense(
            hidden_size, hidden_size, weight_init=TruncatedNormal(initializer_range)
        )

    def construct(
        self, query: ms.Tensor, key: ms.Tensor, value: ms.Tensor, mask: Optional[ms.Tensor] = None
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        batch_size = query.shape[0]

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # multi head
        query = query.view(batch_size, -1, self.num_attention_heads, self.dims_per_head).transpose(
            0, 2, 1, 3
        )
        key = key.view(batch_size, -1, self.num_attention_heads, self.dims_per_head).transpose(
            0, 2, 1, 3
        )
        value = value.view(batch_size, -1, self.num_attention_heads, self.dims_per_head).transpose(
            0, 2, 1, 3
        )

        # self attention
        context, attention = self.attention(query, key, value, attn_mask=mask)
        # concat heads
        context = context.transpose(0, 2, 1, 3).view(batch_size, -1, self.hidden_size)
        output = self.dense(context)

        return output, attention


@constexpr
def generate_factor(dims: int):
    return ms.Tensor(dims, dtype=ms.float32)


class ScaledDotProductAttention(nn.Cell):
    def __init__(self, attention_probs_dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(1.0 - attention_probs_dropout_prob)

    def construct(
        self,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        r"""
        Args:
            query: [batch, num_attention_heads, len_query, dim_query]
            key: [batch, num_attention_heads, len_key, dim_key]
            value: [batch, num_attention_heads, len_value, dim_value]
            attn_mask: [batch, num_attention_heads, len_query, len_key]
        """

        attention = ops.matmul(query, key.transpose(0, 1, 3, 2))
        attention = attention / ops.sqrt(generate_factor(query.shape[-1]))
        if attn_mask is not None:
            attention = attention + attn_mask
        attention = ops.Softmax(axis=-1)(attention)
        attention = self.dropout(attention)
        context = ops.matmul(attention, value)
        return context, attention
