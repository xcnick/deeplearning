import argparse

from transformer.utils.config import Config
from transformer.builder import build_datasets, build_pipelines, build_tf_models

def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument('--train_url', help='the dir to save logs and models')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    model = build_tf_models(cfg.dict["model"])

    train_dataset = build_datasets(cfg.dict["data"]["train"])()

    for pipeline in cfg.dict["train_pipeline"]:
        train_dataset = build_pipelines(pipeline)(train_dataset)

    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    )

    bert_history = model.fit(train_dataset, epochs=10, validation_data=train_dataset)


if __name__ == "__main__":
    main()