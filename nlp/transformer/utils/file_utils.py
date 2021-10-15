import importlib

_torch_available = importlib.util.find_spec("torch") is not None

_tf_available = importlib.util.find_spec("tensorflow") is not None

_ms_available = importlib.util.find_spec("mindspore") is not None

_of_available = importlib.util.find_spec("oneflow") is not None

_pd_available = importlib.util.find_spec("paddle") is not None


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available


def is_ms_available():
    return _ms_available


def is_of_available():
    return _of_available


def is_paddle_available():
    return _pd_available
