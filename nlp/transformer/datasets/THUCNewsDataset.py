import os
from typing import List, Dict

from .CustomDataset import CustomDataset

from transformer.builder import DATASETS
from transformer import builder


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, text, label):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence
            label: (Optional) string. The label of the example. This should be
              specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


@DATASETS.register_module()
class THUCNewsDataset(CustomDataset):

    labels = ["财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]
    label_map = {label: i for i, label in enumerate(labels)}

    def __init__(
        self,
        data_root: str = "",
        mode: str = "train",
        tokenizer_cfg: Dict = {},
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        pad_token: int = 0,
        max_seq_length: int = 512,
        special_tokens_count: int = 2,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.mode = mode
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.max_seq_length = max_seq_length
        self.special_tokens_count = special_tokens_count
        self.num_samples = 0

        self.tokenizer = builder.build_tokenizers(tokenizer_cfg)

    def get_data_dict(self) -> Dict[str, List]:
        examples = self.read_examples_from_files(self.data_root, self.mode)
        dataset_dict = self.convert_examples_to_data_dict(examples)
        return dataset_dict

    def __len__(self) -> int:
        # 需要先调用 get_data_dict
        return self.num_samples

    def read_examples_from_files(self, data_root: str, mode: str) -> List[InputExample]:
        file_path = os.path.join(data_root, "{}.txt".format(mode))
        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                splits = line.strip().split("\t")
                label = splits[0]
                text = splits[-1].replace("\n", "")
                examples.append(
                    InputExample(guid="{}-{}".format(mode, guid_index), text=text, label=label)
                )
                guid_index += 1
        return examples

    def convert_examples_to_data_dict(self, examples: List[InputExample]) -> Dict[str, List]:
        input_id_list = []
        input_mask_list = []
        label_list = []
        for ex_i, example in enumerate(examples):
            if ex_i % 10000 == 0:
                print(f"{ex_i} of {len(examples)}")
            tokens = self.tokenizer.tokenize(example.text)
            if len(tokens) > (self.max_seq_length - self.special_tokens_count):
                tokens = tokens[: (self.max_seq_length - self.special_tokens_count)]
            tokens = [self.cls_token] + tokens + [self.sep_token]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            input_id_list.append(input_ids)
            input_mask_list.append(input_mask)
            label_list.append(self.label_map[example.label])

        self.num_samples = len(input_id_list)
        data_dict = {
            "input_ids": input_id_list,
            "attention_mask": input_mask_list,
            "label_id": label_list,
        }
        return data_dict
