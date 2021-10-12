from .utils.file_utils import is_ms_available, is_torch_available, is_tf_available, is_of_available

from transformer.utils.registry import Registry, build_from_cfg

CONFIGS = Registry("config")
DATASETS = Registry("dataset")
TOKENIZERS = Registry("tokenizer")


def build_config(cfg, default_args=None):
    return build_from_cfg(cfg, registry=CONFIGS, default_args=default_args)


def build_datasets(cfg, default_args=None):
    return build_from_cfg(cfg, registry=DATASETS, default_args=default_args)


def build_tokenizers(cfg, default_args=None):
    return build_from_cfg(cfg, registry=TOKENIZERS, default_args=default_args)


if is_torch_available():
    PT_MODELS = Registry("PT_models")

    def build_torch_models(cfg, default_args=None):
        return build_from_cfg(cfg, registry=PT_MODELS, default_args=default_args)


if is_tf_available():
    TF_MODELS = Registry("tf_models")
    TF_CALLBACKS = Registry("tf_callback")
    TF_OPTIMIZERS = Registry("tf_optimizer")
    TF_TRANSFORMS = Registry("tf_transform")
    TF_PIPELINES = Registry("tf_pipeline")

    def build_tf_pipelines(cfg, default_args=None):
        return build_from_cfg(cfg, registry=TF_PIPELINES, default_args=default_args)

    def build_tf_transforms(cfg, default_args=None):
        return build_from_cfg(cfg, registry=TF_TRANSFORMS, default_args=default_args)

    def build_tf_models(cfg, default_args=None):
        return build_from_cfg(cfg, registry=TF_MODELS, default_args=default_args)

    def build_tf_callbacks(cfg, default_args=None):
        return build_from_cfg(cfg, registry=TF_CALLBACKS, default_args=default_args)

    import tensorflow as tf
    import inspect

    def register_tf_optimizers():
        tf_optimizers = []
        for module_name in dir(tf.keras.optimizers):
            if module_name.startswith("_"):
                continue
            _optim = getattr(tf.keras.optimizers, module_name)
            if inspect.isclass(_optim) and issubclass(_optim, tf.keras.optimizers.Optimizer):
                TF_OPTIMIZERS.register_module()(_optim)
                tf_optimizers.append(module_name)
        return tf_optimizers

    def build_tf_optimizers(cfg, default_args=None):
        if len(TF_OPTIMIZERS) == 0:
            register_tf_optimizers()
        return build_from_cfg(cfg, registry=TF_OPTIMIZERS, default_args=default_args)


if is_ms_available():
    MS_MODELS = Registry("ms_models")

    def build_ms_models(cfg, default_args=None):
        return build_from_cfg(cfg, registry=MS_MODELS, default_args=default_args)


if is_of_available():
    OF_MODELS = Registry("of_models")

    def build_of_models(cfg, default_args=None):
        return build_from_cfg(cfg, registry=OF_MODELS, default_args=default_args)
