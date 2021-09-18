from transformer.utils.registry import Registry, build_from_cfg

CONFIGS = Registry("config")
TF_MODELS = Registry("tf_models")
DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")
TRANSFORMS = Registry("transform")
TOKENIZERS = Registry("tokenizer")


def build_config(cfg, default_args=None):
    return build_from_cfg(cfg, registry=CONFIGS, default_args=default_args)


def build_tf_models(cfg, default_args=None):
    return build_from_cfg(cfg, registry=TF_MODELS, default_args=default_args)


def build_datasets(cfg, default_args=None):
    return build_from_cfg(cfg, registry=DATASETS, default_args=default_args)


def build_pipelines(cfg, default_args=None):
    return build_from_cfg(cfg, registry=PIPELINES, default_args=default_args)


def build_transforms(cfg, default_args=None):
    return build_from_cfg(cfg, registry=TRANSFORMS, default_args=default_args)


def build_tokenizers(cfg, default_args=None):
    return build_from_cfg(cfg, registry=TOKENIZERS, default_args=default_args)
