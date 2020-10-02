import copy

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .config import ConfigBase
from .transformer import AddNormLayer, PositionwiseFeedForward, MultiHeadAttention
from .bert import BertEmbeddings
from .utils import PreTrainedModel, get_activation


class TTAConfig(ConfigBase):
    def __init__(
        self,
        vocab_size,
        embedding_size=128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings


class TTAEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        if position_ids == None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        q_embeddings = position_embeddings + token_type_embeddings
        q_embeddings = self.layer_norm(q_embeddings)
        q_embeddings = self.dropout(q_embeddings)

        return embeddings, q_embeddings


class EncoderTTA(nn.Module):
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
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    EncoderLayerTTA(
                        d_model,
                        nhead,
                        d_feedforward,
                        attention_probs_dropout_prob,
                        hidden_dropout_prob,
                        hidden_act,
                    )
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, x, inputs, mask):
        for layer in self.layers:
            x, attn = layer(x, inputs, mask)
        return x, attn


class EncoderLayerTTA(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        d_feedforward=2048,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        hidden_act="relu",
    ):
        super().__init__()
        self.self = MultiHeadAttention(d_model, nhead, attention_probs_dropout_prob)
        self.feed_forward = PositionwiseFeedForward(d_model, d_feedforward, hidden_dropout_prob, hidden_act)
        self.add_norm = nn.ModuleList(
            [copy.deepcopy(AddNormLayer(d_model, hidden_dropout_prob)) for _ in range(2)]
        )

    def forward(self, x, inputs, mask=None):
        x1, attn = self.self(x, inputs, inputs, mask=mask)
        x = self.add_norm[0](x, x1)
        x1 = self.feed_forward(x)
        x = self.add_norm[1](x, x1)
        return x, attn


class TTAModel(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.embeddings = TTAEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        self.encoder = EncoderTTA(
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            config.hidden_act,
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify input_ids")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        extended_attention_mask = extended_attention_mask.expand(-1, -1, input_shape[-1], -1)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask += torch.eye(input_shape[-1], device=input_ids.device) * -10000.0

        embeddings, q_embeddings = self.embeddings(input_ids, token_type_ids, position_ids)
        if hasattr(self, "embeddings_project"):
            embeddings = self.embeddings_project(embeddings)
            q_embeddings = self.embeddings_project(q_embeddings)

        encoder_outputs, attention_outputs = self.encoder(
            q_embeddings, embeddings, extended_attention_mask
        )
        outputs = (encoder_outputs,)

        return outputs


class TTAMLMHeadModel(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.tta = TTAModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = get_activation(config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self, input_ids, labels=None, attention_mask=None, token_type_ids=None, position_ids=None
    ):
        encoder_outputs = self.tta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )[0]
        hidden_states = self.dense(encoder_outputs)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        mlm_logits = self.decoder(hidden_states)

        mlm_loss = None
        if labels is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            mlm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return (mlm_loss, mlm_logits)
