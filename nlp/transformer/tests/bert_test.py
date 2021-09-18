import pytest

import torch
import transformers
from transformer.transformer.config import ConfigBase
from transformer.transformer.bert import (
    BertForPreTraining,
    load_tf_weights_in_bert,
    load_huggingface_weights_in_bert,
)


class TestBertModel:
    @classmethod
    def setup_class(cls):
        cls.config_file_path = "/workspace/models/nlp/uncased_L-12_H-768_A-12/bert_config.json"
        cls.tf_checkpoint_path = "/workspace/models/nlp/uncased_L-12_H-768_A-12/bert_model.ckpt"
        cls.huggingface_model_path = "/workspace/models/nlp/uncased_L-12_H-768_A-12"
        cls.model_path = "/workspace/models/nlp/uncased_L-12_H-768_A-12/model_pt.bin"
        cls.config = ConfigBase(cls.config_file_path)
        cls.model_tf = BertForPreTraining(cls.config)
        cls.model_hf = BertForPreTraining(cls.config)
        cls.model_base = transformers.BertModel.from_pretrained(
            cls.huggingface_model_path, return_dict=True
        )
        cls.model_base.eval()
        cls.model_base_mlm = transformers.BertForPreTraining.from_pretrained(
            cls.huggingface_model_path, return_dict=True
        )
        cls.model_base_mlm.eval()
        cls.model = BertForPreTraining.from_pretrained(cls.config, cls.model_path)
        cls.model.eval()
        cls.batch_size = 4
        cls.seq_length = 10
        cls.tokens_tensor = torch.randint(
            low=1, high=100, size=(cls.batch_size, cls.seq_length), dtype=torch.long
        )

    @classmethod
    def teardown_class(cls):
        pass

    def test_config(self):
        assert self.config.vocab_size == 30522
        assert self.config.hidden_size == 768
        assert self.config.num_hidden_layers == 12
        assert self.config.num_attention_heads == 12
        assert self.config.intermediate_size == 3072
        assert self.config.hidden_act == "gelu"
        assert self.config.hidden_dropout_prob - 0.1 < 1e-5
        assert self.config.attention_probs_dropout_prob - 0.1 < 1e-5
        assert self.config.max_position_embeddings == 512
        assert self.config.type_vocab_size == 2
        assert self.config.initializer_range - 0.02 < 1e-5

    def test_model(self):
        encoder_outputs, pooled_outputs, prediction_scores = self.model(self.tokens_tensor)
        assert encoder_outputs.shape == (self.batch_size, self.seq_length, self.config.hidden_size)
        assert pooled_outputs.shape == (self.batch_size, self.config.hidden_size)
        assert prediction_scores.shape == (self.batch_size, self.seq_length, self.config.vocab_size)

    def test_tf_and_huggingface_compare(self):
        load_tf_weights_in_bert(self.model_tf, self.config, self.tf_checkpoint_path, with_mlm=True)
        self.model_tf.eval()

        load_huggingface_weights_in_bert(
            self.model_hf, self.config, self.huggingface_model_path, with_mlm=True
        )
        self.model_hf.eval()

        for tf_param, hf_param in zip(self.model_tf.state_dict(), self.model_hf.state_dict()):
            assert torch.equal(
                self.model_tf.state_dict()[tf_param], self.model_hf.state_dict()[hf_param]
            )

        for tf_param, pt_param in zip(self.model_tf.state_dict(), self.model.state_dict()):
            assert torch.equal(
                self.model_tf.state_dict()[tf_param], self.model.state_dict()[pt_param]
            )

    def test_model_forward(self):
        load_tf_weights_in_bert(self.model_tf, self.config, self.tf_checkpoint_path, with_mlm=True)
        self.model_tf.eval()

        load_huggingface_weights_in_bert(
            self.model_hf, self.config, self.huggingface_model_path, with_mlm=True
        )
        self.model_hf.eval()

        tf_encoder_output, tf_pooled_output, tf_mlm_output = self.model_tf(self.tokens_tensor)
        hf_encoder_output, hf_pooled_output, hf_mlm_output = self.model_hf(self.tokens_tensor)
        encoder_output, pooled_output, mlm_output = self.model(self.tokens_tensor)
        base_output = self.model_base(self.tokens_tensor)
        base_mlm_output = self.model_base_mlm(self.tokens_tensor)

        assert torch.max(torch.abs(hf_encoder_output - base_output["last_hidden_state"])) < 1e-5
        assert torch.max(torch.abs(hf_pooled_output - base_output["pooler_output"])) < 1e-5
        assert torch.max(torch.abs(hf_mlm_output - base_mlm_output["prediction_logits"])) < 1e-4

        assert torch.max(torch.abs(tf_encoder_output - base_output["last_hidden_state"])) < 1e-5
        assert torch.max(torch.abs(tf_pooled_output - base_output["pooler_output"])) < 1e-5
        assert torch.max(torch.abs(tf_mlm_output - base_mlm_output["prediction_logits"])) < 1e-4

        assert torch.max(torch.abs(encoder_output - base_output["last_hidden_state"])) < 1e-5
        assert torch.max(torch.abs(pooled_output - base_output["pooler_output"])) < 1e-5
        assert torch.max(torch.abs(mlm_output - base_mlm_output["prediction_logits"])) < 1e-4
