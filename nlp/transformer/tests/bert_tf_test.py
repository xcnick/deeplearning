import pytest

import tensorflow as tf
import torch
import transformers
from transformer.bert_tf import (
    BertConfig,
    TFBertForPreTraining,
    load_tf_weights_in_bert_to_tf,
    load_huggingface_weights_in_bert_to_tf,
)

gpus = tf.config.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)


class TestTFBertModel:
    @classmethod
    def setup_class(cls):
        cls.config_file_path = "/workspace/models/nlp/uncased_L-12_H-768_A-12/bert_config.json"
        cls.tf_checkpoint_path = "/workspace/models/nlp/uncased_L-12_H-768_A-12/bert_model.ckpt"
        cls.huggingface_model_path = "/workspace/models/nlp/uncased_L-12_H-768_A-12"
        cls.model_path = "/workspace/models/nlp/uncased_L-12_H-768_A-12/model_tf.bin"
        cls.config = BertConfig.from_json_file(cls.config_file_path)
        cls.model_tf = TFBertForPreTraining(cls.config)
        cls.model_hf = TFBertForPreTraining(cls.config)
        cls.model_base = transformers.BertModel.from_pretrained(
            cls.huggingface_model_path, return_dict=True
        )
        cls.model_base.eval()
        cls.model_base_mlm = transformers.BertForPreTraining.from_pretrained(
            cls.huggingface_model_path, return_dict=True
        )
        cls.model_base_mlm.eval()
        cls.model = TFBertForPreTraining.from_pretrained(cls.config, cls.model_path)
        cls.batch_size = 4
        cls.seq_length = 10
        cls.tokens_tensor = {
            "input_ids": tf.random.uniform(shape=(4, 10), minval=1, maxval=10, dtype=tf.int32),
            "attention_mask": tf.random.uniform(shape=(4, 10), minval=0, maxval=1, dtype=tf.int32),
            "token_type_ids": tf.random.uniform(shape=(4, 10), minval=0, maxval=1, dtype=tf.int32),
            "position_ids": tf.random.uniform(shape=(4, 10), minval=1, maxval=10, dtype=tf.int32),
        }

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
        self.model_tf(self.tokens_tensor)
        load_tf_weights_in_bert_to_tf(
            self.model_tf, self.config, self.tf_checkpoint_path, with_mlm=True
        )

        self.model_hf(self.tokens_tensor)
        load_huggingface_weights_in_bert_to_tf(
            self.model_hf, self.config, self.huggingface_model_path, with_mlm=True
        )

        for tf_param, pt_param in zip(self.model_tf.variables, self.model_hf.variables):
            assert tf.reduce_all(tf.equal(tf_param.value(), pt_param.value()))

        for tf_param, pt_param in zip(self.model_tf.variables, self.model.variables):
            assert tf.reduce_all(tf.equal(tf_param.value(), pt_param.value()))

    def test_model_forward(self):
        self.model_tf(self.tokens_tensor)
        load_tf_weights_in_bert_to_tf(
            self.model_tf, self.config, self.tf_checkpoint_path, with_mlm=True
        )

        self.model_hf(self.tokens_tensor)
        load_huggingface_weights_in_bert_to_tf(
            self.model_hf, self.config, self.huggingface_model_path, with_mlm=True
        )

        tf_encoder_output, tf_pooled_output, tf_mlm_output = self.model_tf(self.tokens_tensor)
        hf_encoder_output, hf_pooled_output, hf_mlm_output = self.model_hf(self.tokens_tensor)
        encoder_output, pooled_output, mlm_output = self.model(self.tokens_tensor)
        base_output = self.model_base(
            torch.tensor(self.tokens_tensor["input_ids"].numpy(), dtype=torch.long),
            torch.tensor(self.tokens_tensor["attention_mask"].numpy(), dtype=torch.long),
            torch.tensor(self.tokens_tensor["token_type_ids"].numpy(), dtype=torch.long),
            torch.tensor(self.tokens_tensor["position_ids"].numpy(), dtype=torch.long),
        )
        base_mlm_output = self.model_base_mlm(
            torch.tensor(self.tokens_tensor["input_ids"].numpy(), dtype=torch.long),
            torch.tensor(self.tokens_tensor["attention_mask"].numpy(), dtype=torch.long),
            torch.tensor(self.tokens_tensor["token_type_ids"].numpy(), dtype=torch.long),
            torch.tensor(self.tokens_tensor["position_ids"].numpy(), dtype=torch.long),
        )

        last_hidden_state = base_output["last_hidden_state"].detach().numpy()
        pooler_output = base_output["pooler_output"].detach().numpy()
        prediction_logits = base_mlm_output["prediction_logits"].detach().numpy()

        assert tf.reduce_max(tf.abs(hf_encoder_output - last_hidden_state)) < 1e-3
        assert tf.reduce_max(tf.abs(hf_pooled_output - pooler_output)) < 1e-3
        assert tf.reduce_max(tf.abs(hf_mlm_output - prediction_logits)) < 1e-2

        assert tf.reduce_max(tf.abs(tf_encoder_output - last_hidden_state)) < 1e-3
        assert tf.reduce_max(tf.abs(tf_pooled_output - pooler_output)) < 1e-3
        assert tf.reduce_max(tf.abs(tf_mlm_output - prediction_logits)) < 1e-2

        assert tf.reduce_max(tf.abs(encoder_output - last_hidden_state)) < 1e-3
        assert tf.reduce_max(tf.abs(pooled_output - pooler_output)) < 1e-3
        assert tf.reduce_max(tf.abs(mlm_output - prediction_logits)) < 1e-2
