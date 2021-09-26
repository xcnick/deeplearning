from ..utils.file_utils import is_torch_available, is_tf_available

if is_tf_available():
    from .model_tf import TFBertForSequenceClassification