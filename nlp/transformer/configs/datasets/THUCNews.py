dataset_type = "THUCNewsDataset"
data_root = "/workspace/data/THUCNews/"
vocab_file = "/workspace/models/nlp/chinese_wwm_ext/vocab.txt"
tokenizer_cfg = dict(
    type="Tokenizer", vocab_file=vocab_file, do_lower_case=True)

data = dict(
    samples_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        mode="train",
        tokenizer_cfg=tokenizer_cfg),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        mode="val",
        tokenizer_cfg=tokenizer_cfg),
)

train_pipeline = [
    dict(type="Dict2Dataset"),
    dict(type="Map", transforms=[
        dict(type="Identity"),
    ]),
    dict(type="Shuffle", buffer_size=10000),
    dict(type='Repeat'),
    dict(type="PaddedBatch", batch_size=None),
    dict(type="Prefetch"),
]

val_pipeline = [
    dict(type="Dict2Dataset"),
    dict(type="Map", transforms=[
        dict(type="Identity"),
    ]),
    dict(type='Repeat'),
    dict(type="PaddedBatch", batch_size=None),
    dict(type="Prefetch"),
]