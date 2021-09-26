from ..utils.file_utils import is_torch_available, is_tf_available

if is_tf_available():
    from .logger_tf import TextLogger, TensorBoardLogger
    from .modelcheckpoint_tf import ModelCheckpoint
    from .lrscheduler_tf import PolyLRScheduler