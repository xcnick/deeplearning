__version__ = "0.0.1"

from ..utils.file_utils import is_ms_available, is_torch_available, is_tf_available

from .config import ConfigBase
from .tokenizer import Tokenizer

if is_tf_available():
    from .bert_tf import TFBertModel, TFBertForPreTraining

if is_ms_available():
    from .bert_ms import MSBertModel, MSBertForPreTraining
