import copy

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .config import ConfigBase
from .transformer import (
    Encoder,
    PositionwiseFeedForward,
    ScaledDotProductAttention,
)
from .utils import PreTrainedModel, get_activation


class GPT2Config(ConfigBase):
    def __init__(
        self,
        vocab_size,
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
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings


class GPT2Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids == None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class EncoderGPT2(nn.Module):
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
                    EncoderLayerGPT2(
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

    def forward(self, x, mask, pasts):
        presents = []
        for layer, past in zip(self.layers, pasts):
            x, present, attn = layer(x, past, mask)
            presents.append(present)
        return x, attn, presents


class EncoderLayerGPT2(nn.Module):
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
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_feedforward, hidden_dropout_prob, hidden_act
        )
        self.layer_norms = nn.ModuleList([copy.deepcopy(nn.LayerNorm(d_model)) for _ in range(2)])

    def forward(self, x, past, mask=None):
        x1, presents, attn = self.self(x, x, x, layer_past=past, mask=mask)
        x = x + x1
        x1 = self.layer_norms[0](x)
        x1 = self.feed_forward(x1)
        x = x + x1
        x = self.layer_norms[1](x)
        return x, presents, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_head=8, attention_probs_dropout_prob=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.n_head = n_head
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(attention_probs_dropout_prob)
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def forward(self, query, key, value, layer_past=None, mask=None):
        batch_size = query.size(0)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # multi head
        query = query.view(batch_size, -1, self.n_head, self.d_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_head, self.d_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_head, self.d_per_head).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            # print(past_key.shape, key.shape)
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key, value))

        # self attention
        context, attention = self.attention(query, key, value, attn_mask=mask)
        # concat heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(context)

        return output, present, attention


class GPT2Model(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.embeddings = GPT2Embeddings(config)
        self.encoder = EncoderGPT2(
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.attention_probs_dropout_prob,
            config.hidden_dropout_prob,
            config.hidden_act,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * self.config.num_hidden_layers
        else:
            past_length = past[0][0].size(-2)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify input_ids")

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_ids.size(-1) + past_length,
                dtype=torch.long,
                device=input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embeddings = self.embeddings(input_ids, token_type_ids, position_ids)

        if attention_mask is not None:
            assert input_shape[0] > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(input_shape[0], -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        encoder_outputs, attention_outputs, presents = self.encoder(
            embeddings, mask=attention_mask, pasts=past
        )
        encoder_outputs = self.layer_norm(encoder_outputs)

        return encoder_outputs, presents


class GPT2Decoder(nn.Module):
    def __init__(self, config, embedding_weight):
        super().__init__()
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = embedding_weight

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class GPT2LMHeadModel(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2Decoder(config, self.transformer.embeddings.token_embeddings.weight)

    def forward(
        self,
        input_ids,
        labels=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        past=None,
    ):
        encoder_outputs, presents = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            past=past,
        )
        lm_logits = self.lm_head(encoder_outputs)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, lm_logits, presents)


def load_tf_weights_in_gpt2(model, config, tf_checkpoint_path):
    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        raise ImportError("cannot import tensorflow, please install tensorflow first")

    tf_model = tf.train.load_checkpoint(tf_checkpoint_path)

    def _load_tf_variable(key):
        return tf_model.get_tensor(key).squeeze()

    def _load_torch_weight(param, data):
        assert param.shape == data.shape
        param.data = torch.from_numpy(data)

    def _load_embedding(embedding, embedding_path):
        embedding_weight = _load_tf_variable(embedding_path)
        _load_torch_weight(embedding.weight, embedding_weight)

    def _load_layer_norm(layer_norm, layer_norm_base):
        layer_norm_gamma = _load_tf_variable(f"{layer_norm_base}/gamma")
        layer_norm_beta = _load_tf_variable(f"{layer_norm_base}/beta")
        _load_torch_weight(layer_norm.weight, layer_norm_gamma)
        _load_torch_weight(layer_norm.bias, layer_norm_beta)

    def _load_linear(linear, linear_path):
        linear_weight = _load_tf_variable(f"{linear_path}/kernel")
        linear_weight = np.transpose(linear_weight)
        linear_bias = _load_tf_variable(f"{linear_path}/bias")
        _load_torch_weight(linear.weight, linear_weight)
        _load_torch_weight(linear.bias, linear_bias)

    def _load_self_attention(attention, attention_path):
        query_weight = _load_tf_variable(f"{attention_path}/query_layer/kernel")
        key_weight = _load_tf_variable(f"{attention_path}/key_layer/kernel")
        value_weight = _load_tf_variable(f"{attention_path}/value_layer/kernel")

        query_weight = np.transpose(query_weight)
        key_weight = np.transpose(key_weight)
        value_weight = np.transpose(value_weight)

        query_bias = _load_tf_variable(f"{attention_path}/query_layer/bias")
        key_bias = _load_tf_variable(f"{attention_path}/key_layer/bias")
        value_bias = _load_tf_variable(f"{attention_path}/value_layer/bias")

        _load_torch_weight(attention.query.weight, query_weight)
        _load_torch_weight(attention.key.weight, key_weight)
        _load_torch_weight(attention.value.weight, value_weight)

        _load_torch_weight(attention.query.bias, query_bias)
        _load_torch_weight(attention.key.bias, key_bias)
        _load_torch_weight(attention.value.bias, value_bias)

    # loading embedding layer
    _load_embedding(model.embeddings.token_embeddings, "newslm/embeddings/word_embed")
    _load_embedding(
        model.embeddings.position_embeddings, "newslm/embeddings/pos_embed",
    )
    _load_layer_norm(model.embeddings.layer_norm, "newslm/embeddings/LayerNorm_embed_norm")

    # loading transformer encoders
    for layer_idx in range(config.num_hidden_layers):
        encoder_layer = model.encoder.layers[layer_idx]
        encoder_path = f"newslm/layer{layer_idx:02}"

        _load_self_attention(encoder_layer.self, encoder_path)
        _load_linear(encoder_layer.self.dense, f"{encoder_path}/context_projection_layer")
        _load_layer_norm(
            encoder_layer.layer_norms[0], f"{encoder_path}/LayerNorm_mlp_ln0",
        )

        _load_linear(
            encoder_layer.feed_forward.intermediate, f"{encoder_path}/intermediate",
        )
        _load_linear(encoder_layer.feed_forward.output, f"{encoder_path}/output")
        _load_layer_norm(
            encoder_layer.layer_norms[1], f"{encoder_path}/LayerNorm_mlp_ln1",
        )
