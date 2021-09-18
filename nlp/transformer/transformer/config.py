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

    # @classmethod
    # def from_json_file(cls, json_file: str) -> "ConfigBase":
    #     with open(json_file, "r", encoding="utf-8") as reader:
    #         text = reader.read()
    #     return cls(**json.loads(text))


# @CONFIG.register_module()
# class BertConfig(ConfigBase):
#     def __init__(
#         self,
#         json_file: str,
#         # vocab_size: int = 30522,
#         # hidden_size: int = 768,
#         # num_hidden_layers: int = 12,
#         # num_attention_heads: int = 12,
#         # intermediate_size: int = 3072,
#         # hidden_act: str = "gelu",
#         # hidden_dropout_prob: float = 0.1,
#         # attention_probs_dropout_prob: float = 0.1,
#         # max_position_embeddings: int = 512,
#         # type_vocab_size: int = 16,
#         # initializer_range: float = 0.02,
#         **kwargs,
#     ) -> None:
#         super().__init__(**kwargs)
#         #BertConfig.from_json_file(json_file)
#         # self.vocab_size = vocab_size
#         # self.hidden_size = hidden_size
#         # self.num_hidden_layers = num_hidden_layers
#         # self.num_attention_heads = num_attention_heads
#         # self.intermediate_size = intermediate_size
#         # self.hidden_act = hidden_act
#         # self.hidden_dropout_prob = hidden_dropout_prob
#         # self.attention_probs_dropout_prob = attention_probs_dropout_prob
#         # self.max_position_embeddings = max_position_embeddings
#         # self.type_vocab_size = type_vocab_size
#         # self.initializer_range = initializer_range
