__version__ = "0.0.1"

from ..utils.file_utils import is_tf_available, is_torch_available, is_ms_available, is_of_available

from .config import ConfigBase
from .tokenizer import Tokenizer

if is_tf_available():
    from .bert_tf import TFBertModel, TFBertForPreTraining

if is_torch_available():
    from .bert import BertModel, BertForPreTraining

if is_ms_available():
    from .bert_ms import MSBertModel, MSBertForPreTraining

if is_of_available():
    from .bert_of import OFBertModel, OFBertForPreTraining
