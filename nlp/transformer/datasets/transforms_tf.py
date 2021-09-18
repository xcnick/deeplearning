from tensorflow.python.data.ops import dataset_ops

from transformer.builder import TRANSFORMS


@TRANSFORMS.register_module()
class Identity(object):
    def __init__(self):
        pass

    def __call__(self, dataset: dataset_ops.DatasetV2):
        return dataset
