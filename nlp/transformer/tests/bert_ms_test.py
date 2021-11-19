import pytest

import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor
import torch
import transformers

from transformer.builder import build_config, build_ms_models

from transformer.transformer.bert_ms import (
    load_tf_weights_in_bert_to_ms,
    load_huggingface_weights_in_bert_to_ms,
)
import mindspore.context as context

# context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class TestBertModel:

    @classmethod
    def setup_class(cls):
        cls.config_file_path = "/workspace/models/nlp/chinese_wwm_ext/bert_config.json"
        cls.tf_checkpoint_path = "/workspace/models/nlp/chinese_wwm_ext/bert_model.ckpt"
        cls.huggingface_model_path = "/workspace/models/nlp/chinese_wwm_ext"
        cls.model_path = "/workspace/models/nlp/chinese_wwm_ext/bert_model_ms.ckpt"
        model_cfg = dict(
            type="MSBertForPreTraining",
            config=dict(type="ConfigBase", json_file=cls.config_file_path),
        )
        cls.config = build_config(model_cfg["config"])
        cls.model_tf = build_ms_models(model_cfg)
        cls.model_hf = build_ms_models(model_cfg)
        cls.model_base = transformers.BertModel.from_pretrained(
            cls.huggingface_model_path, config=cls.config_file_path)
        cls.model_base.eval()
        # cls.model_base_mlm = transformers.BertForPreTraining.from_pretrained(
        #     cls.huggingface_model_path, config=cls.config_file_path
        # )
        # cls.model_base_mlm.eval()
        model_cfg.update({"model_path": cls.model_path})
        cls.model = build_ms_models(model_cfg)
        cls.batch_size = 4
        cls.seq_length = 10
        cls.tokens_tensor = {
            "input_ids":
            ops.uniform(
                shape=(cls.batch_size, cls.seq_length),
                minval=Tensor(1, mindspore.int32),
                maxval=Tensor(100, mindspore.int32),
                dtype=mindspore.int32,
            ),
            "attention_mask":
            ops.uniform(
                shape=(cls.batch_size, cls.seq_length),
                minval=Tensor(0, mindspore.int32),
                maxval=Tensor(2, mindspore.int32),
                dtype=mindspore.int32,
            ),
            "token_type_ids":
            ops.uniform(
                shape=(cls.batch_size, cls.seq_length),
                minval=Tensor(0, mindspore.int32),
                maxval=Tensor(2, mindspore.int32),
                dtype=mindspore.int32,
            ),
            "position_ids":
            ops.uniform(
                shape=(cls.batch_size, cls.seq_length),
                minval=Tensor(0, mindspore.int32),
                maxval=Tensor(cls.seq_length, mindspore.int32),
                dtype=mindspore.int32,
            ),
        }

    @classmethod
    def teardown_class(cls):
        pass

    def test_config(self):
        assert self.config.vocab_size == 21128
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
        output_dict = self.model(self.tokens_tensor)
        encoder_output = output_dict[0]
        pooled_output = output_dict[1]
        prediction_scores = output_dict[2]
        assert encoder_output.shape == (self.batch_size, self.seq_length,
                                        self.config.hidden_size)
        assert pooled_output.shape == (self.batch_size,
                                       self.config.hidden_size)
        assert prediction_scores.shape == (self.batch_size, self.seq_length,
                                           self.config.vocab_size)

    def test_tf_and_huggingface_compare(self):
        load_tf_weights_in_bert_to_ms(
            self.model_tf, self.config, self.tf_checkpoint_path, with_mlm=True)

        load_huggingface_weights_in_bert_to_ms(
            self.model_hf,
            self.config,
            self.huggingface_model_path,
            with_mlm=True)

        for tf_param, hf_param in zip(self.model_tf.parameters_and_names(),
                                      self.model_hf.parameters_and_names()):
            assert tf_param[0] == hf_param[0]
            assert np.array_equal(tf_param[1].asnumpy(), hf_param[1].asnumpy())

        for tf_param, ms_param in zip(self.model_tf.parameters_and_names(),
                                      self.model.parameters_and_names()):
            assert tf_param[0] == ms_param[0]
            assert np.array_equal(tf_param[1].asnumpy(), ms_param[1].asnumpy())

    def test_model_forward(self):
        load_tf_weights_in_bert_to_ms(
            self.model_tf, self.config, self.tf_checkpoint_path, with_mlm=True)

        load_huggingface_weights_in_bert_to_ms(
            self.model_hf,
            self.config,
            self.huggingface_model_path,
            with_mlm=True)

        output_dict_tf = self.model_tf(self.tokens_tensor)
        output_dict_hf = self.model_hf(self.tokens_tensor)
        output_dict = self.model(self.tokens_tensor)

        self.model_base.eval()
        # self.model_base_mlm.eval()

        tf_encoder_output, tf_pooled_output, tf_mlm_output = (
            output_dict_tf[0],
            output_dict_tf[1],
            output_dict_tf[2],
        )
        hf_encoder_output, hf_pooled_output, hf_mlm_output = (
            output_dict_hf[0],
            output_dict_hf[1],
            output_dict_hf[2],
        )
        encoder_output, pooled_output, mlm_output = (
            output_dict[0],
            output_dict[1],
            output_dict[2],
        )

        base_output = self.model_base(
            input_ids=torch.tensor(
                self.tokens_tensor["input_ids"].asnumpy(), dtype=torch.long),
            attention_mask=torch.tensor(
                self.tokens_tensor["attention_mask"].asnumpy(),
                dtype=torch.long),
            token_type_ids=torch.tensor(
                self.tokens_tensor["token_type_ids"].asnumpy(),
                dtype=torch.long),
            position_ids=torch.tensor(
                self.tokens_tensor["position_ids"].asnumpy(),
                dtype=torch.long),
            return_dict=True,
        )
        # base_mlm_output = self.model_base_mlm(
        #     input_ids=torch.tensor(self.tokens_tensor["input_ids"].asnumpy(), dtype=torch.long),
        #     attention_mask=torch.tensor(
        #         self.tokens_tensor["attention_mask"].asnumpy(), dtype=torch.long
        #     ),
        #     token_type_ids=torch.tensor(
        #         self.tokens_tensor["token_type_ids"].asnumpy(), dtype=torch.long
        #     ),
        #     position_ids=torch.tensor(
        #         self.tokens_tensor["position_ids"].asnumpy(), dtype=torch.long
        #     ),
        #     return_dict=True,
        # )

        last_hidden_state = base_output["last_hidden_state"].detach().numpy()
        pooler_output = base_output["pooler_output"].detach().numpy()
        # prediction_logits = base_mlm_output["prediction_logits"].detach().numpy()

        assert np.max(
            np.abs(hf_encoder_output.asnumpy() -
                   tf_encoder_output.asnumpy())) < 1e-3
        assert np.max(
            np.abs(hf_pooled_output.asnumpy() -
                   tf_pooled_output.asnumpy())) < 1e-3
        assert np.max(
            np.abs(hf_mlm_output.asnumpy() - tf_mlm_output.asnumpy())) < 1e-3

        assert np.max(
            np.abs(hf_encoder_output.asnumpy() -
                   encoder_output.asnumpy())) < 1e-3
        assert np.max(
            np.abs(hf_pooled_output.asnumpy() -
                   pooled_output.asnumpy())) < 1e-3
        assert np.max(
            np.abs(hf_mlm_output.asnumpy() - mlm_output.asnumpy())) < 1e-3

        assert np.max(
            np.abs(hf_encoder_output.asnumpy() - last_hidden_state)) < 1e-1
        assert np.max(
            np.abs(hf_pooled_output.asnumpy() - pooler_output)) < 1e-1

        assert np.max(
            np.abs(tf_encoder_output.asnumpy() - last_hidden_state)) < 1e-1
        assert np.max(
            np.abs(tf_pooled_output.asnumpy() - pooler_output)) < 1e-1

        assert np.max(
            np.abs(encoder_output.asnumpy() - last_hidden_state)) < 1e-1
        assert np.max(np.abs(pooled_output.asnumpy() - pooler_output)) < 1e-1
