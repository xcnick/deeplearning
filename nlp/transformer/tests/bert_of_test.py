import pytest

import numpy as np
import oneflow as flow
import torch
import transformers
from transformer.builder import build_config, build_of_models
from transformer.transformer.bert_of import (
    load_tf_weights_in_bert,
    load_huggingface_weights_in_bert,
)


class TestBertModel:
    @classmethod
    def setup_class(cls):
        cls.config_file_path = "/workspace/models/nlp/chinese_wwm_ext/bert_config.json"
        cls.tf_checkpoint_path = "/workspace/models/nlp/chinese_wwm_ext/bert_model.ckpt"
        cls.huggingface_model_path = "/workspace/models/nlp/chinese_wwm_ext"
        cls.model_path = "/workspace/models/nlp/chinese_wwm_ext/oneflow"
        model_cfg = dict(
            type="OFBertForPreTraining",
            config=dict(type="ConfigBase", json_file=cls.config_file_path),
        )
        cls.config = build_config(model_cfg["config"])
        cls.model_tf = build_of_models(model_cfg)
        cls.model_hf = build_of_models(model_cfg)
        cls.model_base = transformers.BertModel.from_pretrained(
            cls.huggingface_model_path, return_dict=True
        )
        cls.model_base.eval()
        cls.model_base_mlm = transformers.BertForPreTraining.from_pretrained(
            cls.huggingface_model_path, return_dict=True
        )
        cls.model_base_mlm.eval()
        model_cfg.update({"model_path": cls.model_path})
        cls.model = build_of_models(model_cfg)
        cls.model.eval()
        cls.batch_size = 4
        cls.seq_length = 10
        cls.tokens_tensor = {
            "input_ids": flow.randint(
                low=1, high=100, size=(cls.batch_size, cls.seq_length), dtype=flow.long
            ),
            "attention_mask": flow.randint(
                low=0, high=1, size=(cls.batch_size, cls.seq_length), dtype=flow.long
            ),
            "token_type_ids": flow.randint(
                low=0, high=1, size=(cls.batch_size, cls.seq_length), dtype=flow.long
            ),
            "position_ids": flow.randint(
                low=0, high=cls.batch_size, size=(cls.batch_size, cls.seq_length), dtype=flow.long
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
        encoder_output = output_dict["encoder_output"]
        pooled_output = output_dict["pooled_output"]
        prediction_scores = output_dict["prediction_scores"]
        assert encoder_output.shape == (self.batch_size, self.seq_length, self.config.hidden_size)
        assert pooled_output.shape == (self.batch_size, self.config.hidden_size)
        assert prediction_scores.shape == (self.batch_size, self.seq_length, self.config.vocab_size)

    def test_tf_and_huggingface_compare(self):
        load_tf_weights_in_bert(self.model_tf, self.config, self.tf_checkpoint_path, with_mlm=True)
        self.model_tf.eval()

        load_huggingface_weights_in_bert(
            self.model_hf, self.config, self.huggingface_model_path, with_mlm=True
        )
        self.model_hf.eval()

        for key in self.model_tf.state_dict():
            assert np.array_equal(
                self.model_tf.state_dict()[key].numpy(), self.model_hf.state_dict()[key].numpy()
            )

    def test_model_forward(self):
        load_tf_weights_in_bert(self.model_tf, self.config, self.tf_checkpoint_path, with_mlm=True)
        self.model_tf.eval()

        load_huggingface_weights_in_bert(
            self.model_hf, self.config, self.huggingface_model_path, with_mlm=True
        )
        self.model_hf.eval()

        output_dict_tf = self.model_tf(self.tokens_tensor)
        output_dict_hf = self.model_hf(self.tokens_tensor)
        output_dict = self.model(self.tokens_tensor)
        tf_encoder_output, tf_pooled_output, tf_mlm_output = (
            output_dict_tf["encoder_output"],
            output_dict_tf["pooled_output"],
            output_dict_tf["prediction_scores"],
        )
        hf_encoder_output, hf_pooled_output, hf_mlm_output = (
            output_dict_hf["encoder_output"],
            output_dict_hf["pooled_output"],
            output_dict_hf["prediction_scores"],
        )
        encoder_output, pooled_output, mlm_output = (
            output_dict["encoder_output"],
            output_dict["pooled_output"],
            output_dict["prediction_scores"],
        )

        base_output = self.model_base(
            input_ids=torch.tensor(self.tokens_tensor["input_ids"].numpy(), dtype=torch.long),
            attention_mask=torch.tensor(
                self.tokens_tensor["attention_mask"].numpy(), dtype=torch.long
            ),
            token_type_ids=torch.tensor(
                self.tokens_tensor["token_type_ids"].numpy(), dtype=torch.long
            ),
            position_ids=torch.tensor(self.tokens_tensor["position_ids"].numpy(), dtype=torch.long),
        )
        # base_mlm_output = self.model_base_mlm(
        #     input_ids=torch.tensor(self.tokens_tensor["input_ids"].numpy(), dtype=torch.long),
        #     attention_mask=torch.tensor(
        #         self.tokens_tensor["attention_mask"].numpy(), dtype=torch.long
        #     ),
        #     token_type_ids=torch.tensor(
        #         self.tokens_tensor["token_type_ids"].numpy(), dtype=torch.long
        #     ),
        #     position_ids=torch.tensor(self.tokens_tensor["position_ids"].numpy(), dtype=torch.long),
        # )

        last_hidden_state = base_output["last_hidden_state"].detach().numpy()
        pooler_output = base_output["pooler_output"].detach().numpy()
        # prediction_logits = base_mlm_output["prediction_logits"].detach().numpy()

        assert np.max(np.abs(hf_encoder_output.numpy() - last_hidden_state)) < 1e-3
        assert np.max(np.abs(hf_pooled_output.numpy() - pooler_output)) < 1e-3
        # assert np.max(np.abs(hf_mlm_output.numpy() - prediction_logits)) < 1e-3

        assert np.max(np.abs(tf_encoder_output.numpy() - last_hidden_state)) < 1e-3
        assert np.max(np.abs(tf_pooled_output.numpy() - pooler_output)) < 1e-3
        # assert np.max(np.abs(tf_mlm_output.numpy() - prediction_logits)) < 1e-3

        assert np.max(np.abs(encoder_output.numpy() - last_hidden_state)) < 1e-3
        assert np.max(np.abs(pooled_output.numpy() - pooler_output)) < 1e-3
        # assert np.max(np.abs(mlm_output.numpy() - prediction_logits)) < 1e-3
