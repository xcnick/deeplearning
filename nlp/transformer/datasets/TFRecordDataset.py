import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops

from typing import Union, List

from transformer.builder import DATASETS


@DATASETS.register_module()
class TFRecordDataset:

    def __init__(
        self,
        filenames: Union[List[str], str],
        compression_type: str = None,
        buffer_size: int = None,
        num_parallel_reads=None,
    ):
        super().__init__()
        self.filenames = filenames
        self.compression_type = compression_type
        self.buffer_size = buffer_size
        self.num_parallel_reads = num_parallel_reads

    def __call__(self) -> dataset_ops.DatasetV2:
        dataset = tf.data.TFRecordDataset(
            filenames=self.filenames,
            compression_type=self.compression_type,
            buffer_size=self.buffer_size,
            num_parallel_reads=self.num_parallel_reads,
        )
        return dataset
