callbacks = [
    dict(
        type="PolyLRScheduler", total_steps=None, base_lr=0.00001, warmup_iters=100, warmup_lr=1e-6
    ),
    dict(type="TextLogger", print_freq=50),
    dict(type="TensorBoardLogger", log_dir="./tensorboard", update_freq="epoch"),
    dict(
        type="ModelCheckpoint",
        filepath="./model_{epoch:02d}.ckpt",
        monitor="val_sparse_categorical_accuracy",
        save_best_only=True,
        mode="max",
        save_weights_only=True,
        save_freq="epoch",
    ),
]

optimizer = dict(type="Adam", learning_rate=0.00001)

train_cfg = dict(epochs=10, use_keras_progbar=0, steps_per_epoch=0, val_steps=0)
