import json


class ConfigBase(object):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                raise err

    @classmethod
    def from_json_file(cls, json_file: str) -> "ConfigBase":
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls(**json.loads(text))
