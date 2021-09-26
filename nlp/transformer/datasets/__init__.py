from ..utils.file_utils import is_torch_available, is_tf_available

from .THUCNewsDataset import THUCNewsDataset

if is_tf_available():
    from .pipelines_tf import *
    from .transforms_tf import *
