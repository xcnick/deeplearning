dataset_type = "TFRecordDataset"
data_root = "/workspace/data/THUCNews/tfrecord/"

data = dict(
    type=dataset_type,
    samples_per_gpu=16,
    num_train_samples=68578,
    num_val_samples=6500,
    train=dict(type=dataset_type, filenames=[data_root + "train.tfrecord",]),
    val=dict(type=dataset_type, filenames=[data_root + "val.tfrecord",]),
)

train_pipeline = [
    dict(type="Shuffle", buffer_size=10000),
    dict(type="Repeat"),
    dict(type="Map", transforms=[dict(type="THUCNewsTFRecordParser"),]),
    dict(type="PaddedBatch", batch_size=None),
    dict(type="Prefetch"),
]


val_pipeline = [
    dict(type="Repeat"),
    dict(type="Map", transforms=[dict(type="THUCNewsTFRecordParser"),]),
    dict(type="PaddedBatch", batch_size=None),
    dict(type="Prefetch"),
]
