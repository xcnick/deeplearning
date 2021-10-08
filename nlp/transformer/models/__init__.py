from ..utils.file_utils import is_torch_available, is_tf_available, is_ms_available

if is_tf_available():
    from .model_tf import TFBertForSequenceClassification

if is_ms_available():
    from .model_ms import MSBertForSequenceClassification
