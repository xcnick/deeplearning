import os
from typing import List, Dict

from .CustomDataset import CustomDataset

from transformer.builder import DATASETS
from transformer import builder

import pandas as pd
import re


@DATASETS.register_module()
class IQIYIDramaNewsDataset(CustomDataset):

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
        dataframe = self.read_data_from_files(self.data_root, self.mode)
        dataset_dict = self.convert_data_to_dict(dataframe)
        return dataset_dict

    def __len__(self) -> int:
        # 需要先调用 get_data_dict
        return self.num_samples

    def read_data_from_files(self, data_root: str, mode: str):
        file_path = os.path.join(data_root, "{}_dataset.tsv".format(mode))
        df = pd.read_table(file_path)

        # 按照 id 字段排序
        # 剧本序号-场景序号-A-文本序号
        df["drama"] = ""
        df["scenario"] = ""
        df["line"] = ""
        for _, row in df.iterrows():
            row["drama"] = row["id"].split("_")[0]
            row["scenario"] = row["id"].split("_")[1]
            row["line"] = int(row["id"].split("_")[3])

        # 排序
        df = df.sort_values(by=["drama", "scenario", "line"])

        return df

    def convert_data_to_dict(self, data) -> Dict[str, List]:

        def convert2token(line: str, character: str):
            tokens = []
            pattern = '\w\d'
            matchs = re.findall(pattern, line)
            words = re.split(pattern, line)
            i = 0
            j = 1
            for text in words:
                if i == len(words) - 1:
                    tokens += self.tokenizer.tokenize(text)
                    break
                if matchs[i] == character:
                    tokens += self.tokenizer.tokenize(text) + ["[unused10]"]
                else:
                    tokens += self.tokenizer.tokenize(text) + [f"[unused1{j}]"]
                    j += 1
                i += 1
            return tokens

        drama_ids = data["drama"].value_counts().keys()

        input_id_list = []
        input_mask_list = []
        cht_index_list = []
        label_list = []
        test_ids_list = []

        for drama_id in drama_ids:
            one_drama_df = data[data["drama"] == drama_id]
            past_content = None
            after_content = None
            for i, one_line in enumerate(one_drama_df.iterrows()):
                one_line = one_line[1]
                if not pd.isnull(one_line["character"]):
                    if (self.mode == "train" and not pd.isnull(
                            one_line["emotions"])) or self.mode == "test":
                        test_ids_list.append(one_line["id"])
                        content = one_line["content"]
                        if one_line["character"] not in content:
                            continue
                        cht_index = content.index(one_line["character"])
                        # 上文（可选）
                        if i - 1 >= 0 and one_line[
                                "scenario"] == one_drama_df.iloc[i - 1, 5]:
                            past_content = one_drama_df.iloc[i - 1, 1]
                        # 下文（可选）
                        if i < len(one_drama_df) - 1 and one_line[
                                "scenario"] == one_drama_df.iloc[i + 1, 5]:
                            after_content = one_drama_df.iloc[i + 1, 1]

                        if past_content is not None and len(
                                past_content + content) < self.max_seq_length:
                            content = past_content + content
                            cht_index += len(past_content) + 1
                        if after_content is not None and len(
                                content + after_content) < self.max_seq_length:
                            content = content + after_content

                        tokens = [self.cls_token] + convert2token(
                            content, one_line["character"]) + [self.sep_token]
                        input_ids = self.tokenizer.convert_tokens_to_ids(
                            tokens)
                        input_mask = [1] * len(input_ids)
                        input_id_list.append(input_ids)
                        input_mask_list.append(input_mask)

                        cht_index_list.append(cht_index)
                        past_content = None
                        after_content = None
                        if self.mode == "train":
                            label_list.append([
                                int(emotion)
                                for emotion in one_line["emotions"].split(",")
                            ])

        self.num_samples = len(input_id_list)
        if self.mode == "train":
            data_dict = {
                "input_ids": input_id_list,
                "attention_mask": input_mask_list,
                "cht_indices": cht_index_list,
                "labels": label_list,
            }
        else:
            data_dict = {
                "input_ids": input_id_list,
                "attention_mask": input_mask_list,
                "cht_indices": cht_index_list,
                "ids": test_ids_list,
            }
        return data_dict
