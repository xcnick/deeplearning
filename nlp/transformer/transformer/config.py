import json

from transformer.builder import CONFIGS


@CONFIGS.register_module()
class ConfigBase(object):

    def __init__(self, json_file: str, **kwargs) -> None:
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        for key, value in dict(json.loads(text), **kwargs).items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                raise err
