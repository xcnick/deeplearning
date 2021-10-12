import math

import oneflow as flow
import oneflow.nn as nn
from oneflow import Tensor
from typing import Tuple, Optional

from .utils_of import get_activation


class OFEncoder(nn.Module):
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
        self.layers = nn.ModuleList(
            [
                OFEncoderLayer(
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

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        for layer in self.layers:
            x, attn = layer(x, mask)
        return x, attn


class OFEncoderLayer(nn.Module):
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
        self.self = OFMultiHeadAttention(d_model, nhead, attention_probs_dropout_prob)
        self.feed_forward = OFPositionwiseFeedForward(
            d_model, d_feedforward, hidden_dropout_prob, hidden_act
        )
        self.add_norm = nn.ModuleList(
            [OFAddNormLayer(d_model, hidden_dropout_prob) for _ in range(2)]
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x1, attn = self.self(x, x, x, mask)
        x = self.add_norm[0](x, x1)
        x1 = self.feed_forward(x)
        x = self.add_norm[1](x, x1)
        return x, attn


class OFAddNormLayer(nn.Module):
    def __init__(self, d_model: int, hidden_dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: Tensor, x1: Tensor) -> Tensor:
        return self.layer_norm(x + self.dropout(x1))


class OFPositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_feedforward: int,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "relu",
    ) -> None:
        super().__init__()
        self.hidden_act = hidden_act
        self.intermediate = nn.Linear(d_model, d_feedforward)
        self.output = nn.Linear(d_feedforward, d_model)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(self.dropout(get_activation(self.hidden_act)(self.intermediate(x))))


class OFMultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int = 512, n_head: int = 8, attention_probs_dropout_prob: float = 0.1
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.n_head = n_head
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.attention = OFScaledDotProductAttention(attention_probs_dropout_prob)
        self.dense = nn.Linear(d_model, d_model)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = query.size(0)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # multi head
        query = query.view(batch_size, -1, self.n_head, self.d_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_head, self.d_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_head, self.d_per_head).transpose(1, 2)

        # self attention
        context, attention = self.attention(query, key, value, attn_mask=mask)
        # concat heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(context)

        return output, attention


class OFScaledDotProductAttention(nn.Module):
    def __init__(self, attention_probs_dropout_prob: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            query: [batch, len_query, dim_query]
            key: [batch, len_key, dim_key]
            value: [batch, len_value, dim_value]
            attn_mask: [batch, len_query, len_key]
        """
        attention = flow.matmul(query, key.transpose(-1, -2))
        attention = attention / math.sqrt(query.size(-1))
        if attn_mask is not None:
            attention = attention + attn_mask
        attention = nn.Softmax(dim=-1)(attention)
        attention = self.dropout(attention)
        context = flow.matmul(attention, value)
        return context, attention
