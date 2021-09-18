from abc import ABCMeta
from typing import Union, Dict, List
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops


from transformer.builder import PIPELINES
from transformer import builder


class DataPipeline(metaclass=ABCMeta):
    def __init__(self):
        pass

    def __call__(self, dataset: Union[Dict, dataset_ops.DatasetV2]):
        # assert isinstance(dataset, dataset_ops.DatasetV2)
        return dataset


@PIPELINES.register_module()
class Dict2Dataset(DataPipeline):
    def __init__(self, keys: List[str]=None):
        super().__init__()
        self.keys = keys

    def __call__(self, dataset: Union[Dict, dataset_ops.DatasetV2]):
        assert isinstance(dataset, Dict)
        dataset_dict = self.convert_dataset(dataset)
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            dataset_dict
        )
        return tf_dataset

    def convert_dataset(self, dataset: Union[Dict, dataset_ops.DatasetV2]):
        if self.keys is None:
            self.keys = dataset.keys()

        dataset_dict = {}
        for key in dataset.keys():
            try:
                dataset_dict[key] = tf.constant(dataset[key])
            except:
                dataset_dict[key] = tf.ragged.constant(dataset[key])

        return dataset_dict

@PIPELINES.register_module()
class Map(DataPipeline):
    def __init__(
        self,
        transforms,
        num_parallel_calls: tf.Tensor = tf.data.AUTOTUNE,
        deterministic: bool = True,
    ):
        super().__init__()

        if not isinstance(transforms, list):
            transforms = [transforms]

        for i in range(len(transforms)):
            if isinstance(transforms[i], dict):
                transforms[i] = builder.build_transforms(transforms[i])

            if not callable(transforms[i]):
                raise TypeError("`Apply` requires each transform to be callable")

        self.transforms = transforms
        self.num_parallel_calls = num_parallel_calls
        self.deterministic = deterministic

    def __call__(self, dataset: dataset_ops.DatasetV2) -> dataset_ops.DatasetV2:
        for transform in self.transforms:
            dataset = dataset.map(
                map_func=transform,
                num_parallel_calls=self.num_parallel_calls,
                deterministic=self.deterministic,
            )
        return dataset


@PIPELINES.register_module()
class Shuffle(DataPipeline):
    def __init__(self, buffer_size: int, seed: int = None, reshuffle_each_iteration: bool = None):
        super().__init__()

        self.buffer_size = buffer_size
        self.seed = seed
        self.reshuffle_each_iteration = reshuffle_each_iteration

    def __call__(self, dataset: dataset_ops.DatasetV2) -> dataset_ops.DatasetV2:
        dataset = dataset.shuffle(
            buffer_size=self.buffer_size,
            seed=self.seed,
            reshuffle_each_iteration=self.reshuffle_each_iteration,
        )
        return dataset


@PIPELINES.register_module()
class Repeat(DataPipeline):
    def __init__(self, count=None):
        super().__init__()

        self.count = count

    def __call__(self, dataset: dataset_ops.DatasetV2) -> dataset_ops.DatasetV2:
        dataset = dataset.repeat(count=self.count)
        return dataset


@PIPELINES.register_module()
class Prefetch(DataPipeline):
    def __init__(self, buffer_size: tf.Tensor = tf.data.AUTOTUNE):
        super().__init__()

        self.buffer_size = buffer_size

    def __call__(self, dataset: dataset_ops.DatasetV2) -> dataset_ops.DatasetV2:
        dataset = dataset.prefetch(buffer_size=self.buffer_size)
        return dataset


@PIPELINES.register_module()
class PaddedBatch(DataPipeline):
    def __init__(
        self,
        batch_size: tf.Tensor,
        padded_shapes: Union[tf.TensorShape, tf.Tensor] = None,
        padding_values: tf.Tensor = None,
        drop_remainder: bool = False,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.padded_shapes = padded_shapes
        self.padding_values = padding_values
        self.drop_remainder = drop_remainder

    def __call__(self, dataset: dataset_ops.DatasetV2) -> dataset_ops.DatasetV2:
        dataset = dataset.padded_batch(
            batch_size=self.batch_size,
            padded_shapes=self.padded_shapes,
            padding_values=self.padding_values,
            drop_remainder=self.drop_remainder,
        )
        return dataset
