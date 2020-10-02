import unittest

import tensorflow as tf
from transformer.bert_tf import (
    BertConfig,
    TFBertForPreTraining,
    load_tf_weights_in_bert_to_tf,
    load_huggingface_weights_in_bert_to_tf,
)


class TestTFBertModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_file_path = "/home/orient/chi/model/uncased_L-12_H-768_A-12/bert_config.json"
        cls.tf_checkpoint_path = "/home/orient/chi/model/uncased_L-12_H-768_A-12/bert_model.ckpt"
        cls.huggingface_model_path = "/home/orient/chi/model/uncased_L-12_H-768_A-12"
        cls.model_path = "/home/orient/chi/models/transformer/bert_model.ckpt"
        cls.config = BertConfig.from_json_file(cls.config_file_path)
        cls.model_tf = TFBertForPreTraining(cls.config)
        cls.model_hf = TFBertForPreTraining(cls.config)
        cls.model = TFBertForPreTraining.from_pretrained(cls.config, cls.model_path)
        cls.batch_size = 4
        cls.seq_length = 10
        cls.tokens_tensor = tf.random.uniform(shape=(4, 10), minval=1, maxval=10, dtype=tf.int32)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_config(self):
        self.assertEqual(self.config.vocab_size, 30522)
        self.assertEqual(self.config.hidden_size, 768)
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
        encoder_outputs, pooled_outputs, prediction_scores = self.model(self.tokens_tensor)
        self.assertEqual(
            encoder_outputs.shape, (self.batch_size, self.seq_length, self.config.hidden_size)
        )
        self.assertEqual(pooled_outputs.shape, (self.batch_size, self.config.hidden_size))
        self.assertEqual(
            prediction_scores.shape, (self.batch_size, self.seq_length, self.config.vocab_size)
        )

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
            self.assertTrue(tf.reduce_all(tf.equal(tf_param.value(), pt_param.value())))

        for tf_param, pt_param in zip(self.model_tf.variables, self.model.variables):
            self.assertTrue(tf.reduce_all(tf.equal(tf_param.value(), pt_param.value())))

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

        self.assertTrue(tf.reduce_max(tf.abs(tf_encoder_output - hf_encoder_output)) < 1e-5)
        self.assertTrue(tf.reduce_max(tf.abs(tf_pooled_output - hf_pooled_output)) < 1e-5)
        self.assertTrue(tf.reduce_max(tf.abs(tf_mlm_output - hf_mlm_output)) < 1e-4)
        self.assertTrue(tf.reduce_max(tf.abs(tf_encoder_output - encoder_output)) < 1e-5)
        self.assertTrue(tf.reduce_max(tf.abs(tf_pooled_output - pooled_output)) < 1e-5)
        self.assertTrue(tf.reduce_max(tf.abs(tf_mlm_output - mlm_output)) < 1e-4)
