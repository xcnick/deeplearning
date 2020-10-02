import unittest

import torch
from transformer.albert import (
    AlbertConfig,
    AlbertForPreTraining,
    load_tf_weights_in_albert,
    load_huggingface_weights_in_albert,
)


class TestAlbertModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_file_path = "/home/orient/chi/model/albert_base/albert_config.json"
        cls.tf_checkpoint_path = "/home/orient/chi/model/albert_base/model.ckpt-best"
        cls.huggingface_model_path = "/home/orient/chi/model/albert_base"
        cls.model_path = "/home/orient/chi/model/albert_base/albert_model_pt.bin"
        cls.config = AlbertConfig.from_json_file(cls.config_file_path)
        cls.model_tf = AlbertForPreTraining(cls.config)
        cls.model_hf = AlbertForPreTraining(cls.config)
        cls.model = AlbertForPreTraining.from_pretrained(cls.config, cls.model_path)
        cls.model.eval()

    @classmethod
    def tearDownClass(cls):
        pass

    def test_config(self):
        self.assertEqual(self.config.vocab_size, 30000)
        self.assertEqual(self.config.hidden_size, 768)
        self.assertEqual(self.config.embedding_size, 128)
        self.assertEqual(self.config.num_hidden_groups, 1)
        self.assertEqual(self.config.inner_group_num, 1)
        self.assertEqual(self.config.num_hidden_layers, 12)
        self.assertEqual(self.config.num_attention_heads, 12)
        self.assertEqual(self.config.intermediate_size, 3072)
        self.assertEqual(self.config.hidden_act, "gelu")
        self.assertEqual(self.config.hidden_dropout_prob, 0.1)
        self.assertEqual(self.config.attention_probs_dropout_prob, 0.1)
        self.assertEqual(self.config.max_position_embeddings, 512)
        self.assertEqual(self.config.type_vocab_size, 2)
        self.assertEqual(self.config.initializer_range, 0.02)

    def test_model(self):
        batch_size = 4
        seq_length = 10
        tokens_tensor = torch.randint(0, 100, (batch_size, seq_length))

        encoder_outputs, pooled_outputs, mlm_outputs = self.model_tf(tokens_tensor)
        self.assertEqual(encoder_outputs.shape, (batch_size, seq_length, self.config.hidden_size))
        self.assertEqual(pooled_outputs.shape, (batch_size, self.config.hidden_size))
        self.assertEqual(mlm_outputs.shape, (batch_size, seq_length, self.config.vocab_size))

    def test_tf_and_huggingface_compare(self):
        load_tf_weights_in_albert(self.model_tf, self.config, self.tf_checkpoint_path)
        self.model_tf.eval()

        load_huggingface_weights_in_albert(self.model_hf, self.config, self.huggingface_model_path)
        self.model_hf.eval()

        for tf_param, pt_param in zip(self.model_tf.state_dict(), self.model_hf.state_dict()):
            self.assertTrue(
                torch.equal(
                    self.model_tf.state_dict()[tf_param], self.model_hf.state_dict()[pt_param]
                )
            )
        for tf_param, pt_param in zip(self.model.state_dict(), self.model_tf.state_dict()):
            self.assertTrue(
                torch.equal(self.model.state_dict()[tf_param], self.model_tf.state_dict()[pt_param])
            )

    def test_model_forward(self):
        load_tf_weights_in_albert(self.model_tf, self.config, self.tf_checkpoint_path)
        self.model_tf.eval()

        load_huggingface_weights_in_albert(self.model_hf, self.config, self.huggingface_model_path)
        self.model_hf.eval()

        input_ids = [101, 2023, 2003, 1037, 7953, 102]

        tokens_tensor = torch.tensor(input_ids).unsqueeze(0)
        tf_encoder_output, tf_pooled_output, tf_mlm_output = self.model_tf(tokens_tensor)
        hf_encoder_output, hf_pooled_output, hf_mlm_output = self.model_hf(tokens_tensor)
        encoder_output, pooled_output, mlm_output = self.model(tokens_tensor)

        self.assertTrue(torch.max(torch.abs(tf_encoder_output - hf_encoder_output)) < 1e-5)
        self.assertTrue(torch.max(torch.abs(tf_pooled_output - hf_pooled_output)) < 1e-5)
        self.assertTrue(torch.max(torch.abs(tf_mlm_output - hf_mlm_output)) < 1e-4)
        self.assertTrue(torch.max(torch.abs(tf_encoder_output - encoder_output)) < 1e-5)
        self.assertTrue(torch.max(torch.abs(tf_pooled_output - pooled_output)) < 1e-5)
        self.assertTrue(torch.max(torch.abs(tf_mlm_output - mlm_output)) < 1e-4)
