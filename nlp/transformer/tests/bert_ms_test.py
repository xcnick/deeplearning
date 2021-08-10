import pytest

import numpy as np
import mindspore
from mindspore import Tensor
import torch
import transformers
from transformer.bert_ms import (
    BertConfig,
    MSBertForPreTraining,
    load_tf_weights_in_bert_to_ms,
    load_huggingface_weights_in_bert_to_ms,
)
import mindspore.context as context

# context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class TestBertModel:
    @classmethod
    def setup_class(cls):
        cls.config_file_path = "/mnt/data/models/nlp/uncased_L-12_H-768_A-12/bert_config.json"
        cls.tf_checkpoint_path = "/mnt/data/models/nlp/uncased_L-12_H-768_A-12/bert_model.ckpt"
        cls.huggingface_model_path = "/mnt/data/models/nlp/uncased_L-12_H-768_A-12"
        cls.model_path = "/mnt/data/models/nlp/uncased_L-12_H-768_A-12/model_ms.ckpt"
        cls.config = BertConfig.from_json_file(cls.config_file_path)
        cls.model_tf = MSBertForPreTraining(cls.config)
        cls.model_hf = MSBertForPreTraining(cls.config)
        cls.model_base = transformers.BertModel.from_pretrained(
            cls.huggingface_model_path, return_dict=True
        )
        cls.model_base.eval()
        cls.model_base_mlm = transformers.BertForPreTraining.from_pretrained(
            cls.huggingface_model_path, return_dict=True
        )
        cls.model_base_mlm.eval()
        cls.model = MSBertForPreTraining.from_pretrained(cls.config, cls.model_path)
        cls.batch_size = 4
        cls.seq_length = 10
        cls.input_tokens = np.random.randint(
            0, 100, (cls.batch_size, cls.seq_length), dtype=np.int32
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
        encoder_outputs, pooled_outputs, prediction_scores = self.model(
            input_ids=Tensor(self.input_tokens, dtype=mindspore.int32)
        )
        assert encoder_outputs.shape == (self.batch_size, self.seq_length, self.config.hidden_size)
        assert pooled_outputs.shape == (self.batch_size, self.config.hidden_size)
        assert prediction_scores.shape == (self.batch_size, self.seq_length, self.config.vocab_size)

    def test_tf_and_huggingface_compare(self):
        load_tf_weights_in_bert_to_ms(
            self.model_tf, self.config, self.tf_checkpoint_path, with_mlm=True
        )

        load_huggingface_weights_in_bert_to_ms(
            self.model_hf, self.config, self.huggingface_model_path, with_mlm=True
        )

        for tf_param, hf_param in zip(
            self.model_tf.parameters_and_names(), self.model_hf.parameters_and_names()
        ):
            assert tf_param[0] == hf_param[0]
            assert np.array_equal(tf_param[1].asnumpy(), hf_param[1].asnumpy())

        for tf_param, ms_param in zip(
            self.model_tf.parameters_and_names(), self.model.parameters_and_names()
        ):
            assert tf_param[0] == ms_param[0]
            assert np.array_equal(tf_param[1].asnumpy(), ms_param[1].asnumpy())

    def test_model_forward(self):
        load_tf_weights_in_bert_to_ms(
            self.model_tf, self.config, self.tf_checkpoint_path, with_mlm=True
        )

        load_huggingface_weights_in_bert_to_ms(
            self.model_hf, self.config, self.huggingface_model_path, with_mlm=True
        )

        tf_encoder_output, tf_pooled_output, tf_mlm_output = self.model_tf(
            Tensor(self.input_tokens, dtype=mindspore.int32)
        )
        hf_encoder_output, hf_pooled_output, hf_mlm_output = self.model_hf(
            Tensor(self.input_tokens, dtype=mindspore.int32)
        )
        encoder_output, pooled_output, mlm_output = self.model(
            Tensor(self.input_tokens, dtype=mindspore.int32)
        )

        base_output = self.model_base(torch.tensor(self.input_tokens, dtype=torch.long))
        base_mlm_output = self.model_base_mlm(torch.tensor(self.input_tokens, dtype=torch.long))

        last_hidden_state = base_output["last_hidden_state"].detach().numpy()
        pooler_output = base_output["pooler_output"].detach().numpy()
        prediction_logits = base_mlm_output["prediction_logits"].detach().numpy()

        assert np.max(np.abs(hf_encoder_output.asnumpy() - last_hidden_state)) < 1e-1
        assert np.max(np.abs(hf_pooled_output.asnumpy() - pooler_output)) < 1e-1
        assert np.max(np.abs(hf_mlm_output.asnumpy() - prediction_logits)) < 1e-1

        assert np.max(np.abs(tf_encoder_output.asnumpy() - last_hidden_state)) < 1e-1
        assert np.max(np.abs(tf_pooled_output.asnumpy() - pooler_output)) < 1e-1
        assert np.max(np.abs(tf_mlm_output.asnumpy() - prediction_logits)) < 1e-1

        assert np.max(np.abs(encoder_output.asnumpy() - last_hidden_state)) < 1e-1
        assert np.max(np.abs(pooled_output.asnumpy() - pooler_output)) < 1e-1
        assert np.max(np.abs(mlm_output.asnumpy() - prediction_logits)) < 1e-1
