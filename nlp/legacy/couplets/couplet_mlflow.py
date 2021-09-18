import os
import pandas as pd
import torch

from transformers import BertConfig, BertTokenizer
from bert_seq_model import BertForSeq2Seq

class Args(object):
    def __init__(
        self,
        model_type,
        model_name_or_path,
        max_seq_length,
        do_lower_case,
        no_cuda,
    ):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        self.no_cuda = no_cuda

root_path = os.path.dirname(os.path.realpath(__file__))

args = Args(
    model_type="bert",
    model_name_or_path="/home/orient/chi/models/couplets/output/",
    max_seq_length=128,
    do_lower_case=True,
    no_cuda=True,
)


class PyTorchCoupletPyfunc(object):
    def __init__(self):
        self.args = args
        self.config = BertConfig.from_pretrained(self.args.model_name_or_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_name_or_path,
                                                       do_lower_case=self.args.do_lower_case)
        self.model = BertForSeq2Seq.from_pretrained(self.args.model_name_or_path,
                                                    config=self.config)

        self.device = torch.device("cpu")
        self.model.to(self.device)

    def predict(self, text):
        input_str = text["input"][0]

        inputs = self.tokenizer.encode_plus(input_str)

        output_ids = self.model.beam_search(input_ids=torch.tensor([inputs["input_ids"]], dtype=torch.long),
                               token_type_ids=torch.tensor([inputs["token_type_ids"]], dtype=torch.long),
                               sep_id=self.tokenizer.sep_token_id,
                               beam_size=3,
                               out_max_length=self.args.max_seq_length//2)
        output_tokens = self.tokenizer.decode(output_ids)

        res = pd.DataFrame(
            {
                "pred": output_tokens,
            },
            index=[0],
        )
        return res


def _load_pyfunc(path):
    return PyTorchCoupletPyfunc()
