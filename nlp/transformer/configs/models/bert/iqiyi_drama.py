_base_ = ["../../datasets/IQIYIDrama.py", "../../runtime/tf_runtime.py"]

model = dict(
    type="TFBertForIQIYIDrama",
    config=dict(
        type="ConfigBase",
        json_file="/workspace/models/nlp/chinese_wwm_ext/bert_config.json",
        num_labels=4,
    ),
    model_path="/workspace/models/nlp/chinese_wwm_ext/model_tf.bin"
)

callbacks = [
    dict(
        type="PolyLRScheduler", total_steps=None, base_lr=0.00001, warmup_iters=10, warmup_lr=1e-6
    ),
    dict(type="TextLogger", print_freq=50),
    dict(type="TensorBoardLogger", log_dir="./tensorboard", update_freq="epoch"),
    dict(
        type="ModelCheckpoint",
        filepath="./model_{epoch:02d}.ckpt",
        monitor="val_score",
        save_best_only=True,
        mode="max",
        save_weights_only=True,
        save_freq="epoch",
    ),
]

train_cfg = dict(epochs=20)
