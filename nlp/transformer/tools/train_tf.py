import argparse
import os
import time

from transformer.utils.config import Config
from transformer.utils.logging import get_root_logger
from transformer.builder import (
    build_datasets,
    build_tf_pipelines,
    build_tf_models,
    build_tf_callbacks,
    build_tf_optimizers,
)

import tensorflow as tf

gpus = tf.config.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer")
    parser.add_argument("--config", type=str, help="train config file path")
    parser.add_argument(
        "--train_url", type=str, help="the dir to save logs and models")
    parser.add_argument(
        "--fp16",
        default=False,
        action="store_true",
        help="whether to use mixed precision")
    args = parser.parse_args()

    return args


def _adjust_lr(cfg: dict, num_replicas: int):
    if "learning_rate" in cfg.dict["optimizer"]:
        cfg.dict["optimizer"]["learning_rate"] *= num_replicas

    callback_list = cfg.dict.get("callbacks")
    if not callback_list:
        return

    for callback in callback_list:
        if callback["type"].endswith("LRScheduler"):
            if "base_lr" in callback:
                callback["base_lr"] *= num_replicas


def _adjust_batchsize(cfg: dict, num_replicas: int, num_train_samples: int,
                      num_val_samples: int):
    samples_per_gpu = cfg.dict["data"]["samples_per_gpu"]
    global_batch_size = num_replicas * samples_per_gpu
    for i in range(len(cfg.dict["train_pipeline"]) - 1, -1, -1):
        if cfg.dict["train_pipeline"][i]["type"].endswith("Batch"):
            cfg.dict["train_pipeline"][i]["batch_size"] = global_batch_size
            break

    if cfg.dict["train_cfg"]["steps_per_epoch"] == 0:
        cfg.dict["train_cfg"][
            "steps_per_epoch"] = num_train_samples // global_batch_size + 1

    # val
    if "val_pipeline" in cfg.dict:
        for i in range(len(cfg.dict["val_pipeline"]) - 1, -1, -1):
            if cfg.dict["val_pipeline"][i]["type"].endswith("Batch"):
                cfg.dict["val_pipeline"][i]["batch_size"] = global_batch_size
                break
        if cfg.dict["train_cfg"]["val_steps"] == 0:
            cfg.dict["train_cfg"][
                "val_steps"] = num_val_samples // global_batch_size + 1


def _adjust_callback(cfg: dict, train_url: str = None):
    callback_list = cfg.dict.get("callbacks")
    if not callback_list:
        return

    for callback in callback_list:
        if callback["type"] == "TextLogger":
            cfg.dict["train_cfg"]["use_keras_progbar"] = 0
            callback.update(
                dict(
                    epochs=cfg.dict["train_cfg"]["epochs"],
                    steps_per_epoch=cfg.dict["train_cfg"]["steps_per_epoch"],
                    val_steps=cfg.dict["train_cfg"]["val_steps"],
                ))
        elif callback["type"] == "TensorBoardLogger":
            callback["log_dir"] = os.path.join(train_url, callback["log_dir"])
        elif callback["type"] == "ModelCheckpoint":
            callback["filepath"] = os.path.join(train_url,
                                                callback["filepath"])
        elif callback["type"].endswith("LRScheduler"):
            callback["total_steps"] = (
                cfg.dict["train_cfg"]["epochs"] *
                cfg.dict["train_cfg"]["steps_per_epoch"])


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    get_root_logger(
        log_file=os.path.join(args.train_url, f"train_{time_str}.log"))

    if args.fp16:
        from tensorflow.keras import mixed_precision

        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

    train_optimizer = build_tf_optimizers(cfg.dict["optimizer"])

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = build_tf_models(cfg.dict["model"])
        model.compile(optimizer=train_optimizer, )

    train_dataset_obj = build_datasets(cfg.dict["data"]["train"])
    val_dataset_obj = build_datasets(cfg.dict["data"]["val"])

    if cfg.dict["dataset_type"] == "TFRecordDataset":
        train_dataset = train_dataset_obj()
        val_dataset = val_dataset_obj()

        num_train_samples = cfg.dict["data"]["num_train_samples"]
        num_val_samples = cfg.dict["data"]["num_val_samples"]
    else:
        train_dataset = train_dataset_obj.get_data_dict()
        val_dataset = val_dataset_obj.get_data_dict()

        num_train_samples = len(train_dataset_obj)
        num_val_samples = len(val_dataset_obj)

    _adjust_batchsize(
        cfg,
        mirrored_strategy.num_replicas_in_sync,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
    )

    _adjust_lr(cfg, num_replicas=mirrored_strategy.num_replicas_in_sync)
    _adjust_callback(cfg, args.train_url)

    for pipeline in cfg.dict["train_pipeline"]:
        train_dataset = build_tf_pipelines(pipeline)(train_dataset)

    for pipeline in cfg.dict["val_pipeline"]:
        val_dataset = build_tf_pipelines(pipeline)(val_dataset)

    callback_list = []
    for callback in cfg.dict["callbacks"]:
        callback_list.append(build_tf_callbacks(callback))

    model.fit(
        x=train_dataset,
        epochs=cfg.dict["train_cfg"]["epochs"],
        steps_per_epoch=cfg.dict["train_cfg"]["steps_per_epoch"],
        validation_data=val_dataset,
        validation_steps=cfg.dict["train_cfg"]["val_steps"],
        callbacks=callback_list,
        verbose=cfg.dict["train_cfg"]["use_keras_progbar"],
    )


if __name__ == "__main__":
    main()
