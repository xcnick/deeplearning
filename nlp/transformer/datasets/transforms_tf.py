import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops

from transformer.builder import TF_TRANSFORMS


@TF_TRANSFORMS.register_module()
class Identity(object):
    def __init__(self):
        pass

    def __call__(self, dataset: dataset_ops.DatasetV2):
        return dataset


@TF_TRANSFORMS.register_module()
class THUCNewsTFRecordParser(object):
    def __init__(self):
        self.feature_description = {
            "input_ids": tf.io.RaggedFeature(tf.int64),
            "attention_mask": tf.io.RaggedFeature(tf.int64),
            "label_id": tf.io.RaggedFeature(tf.int64),
        }

    def __call__(self, example_proto):
        dataset = tf.io.parse_single_example(example_proto, self.feature_description)
        return dataset
