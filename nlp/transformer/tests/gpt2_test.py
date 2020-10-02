import unittest

import torch
import torch.nn.functional as F
from transformer.gpt2 import GPT2Config, GPT2Model, GPT2Decoder, load_tf_weights_in_gpt2
from transformer.tokenizer import Tokenizer

"""
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(
        logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits
    )


def sample_sequence(
    model,
    decoder_model,
    length,
    batch_size=None,
    context=None,
    temperature=1,
    top_k=40,
    device="cuda",
    sample=True,
):
    context = (
        torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    )
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in range(length):
            logits, past = model(prev, past=past)
            logits = decoder_model(logits)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output
"""


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, model_decoder, context, length, temperature=1.0, top_k=30, top_p=0.0, device="cpu"):
    inputs = torch.tensor(context, dtype=torch.long).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context
    with torch.no_grad():
        for i in range(length):
            logits, past = model(prev, past=past)
            logits = model_decoder(logits)
            logits = logits[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    return generate


class TestGPT2Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_file_path = "/home/orient/chi/model/gpt2-new/config.json"
        cls.tf_checkpoint_path = "/home/orient/chi/model/gpt2-new/model.ckpt-220000"
        cls.vocab_file_path = "/home/orient/chi/model/gpt2-new/vocab.txt"
        cls.config = GPT2Config.from_json_file(cls.config_file_path)
        cls.device = "cuda"
        cls.model_tf = GPT2Model(cls.config)
        load_tf_weights_in_gpt2(cls.model_tf, cls.config, cls.tf_checkpoint_path)
        cls.model_tf.to(cls.device)
        cls.model_decoder = GPT2Decoder(
            cls.config, cls.model_tf.embeddings.token_embeddings.weight
        ).to(cls.device)
        cls.model_tf.eval()
        cls.model_decoder.eval()
        cls.tokenizer = Tokenizer(cls.vocab_file_path, do_lower_case=True)  # 建立分词器

    @classmethod
    def tearDownClass(cls):
        pass

    def test_config(self):
        self.assertEqual(self.config.vocab_size, 8021)
        self.assertEqual(self.config.hidden_size, 1536)
        self.assertEqual(self.config.num_hidden_layers, 48)
        self.assertEqual(self.config.num_attention_heads, 24)
        self.assertEqual(self.config.intermediate_size, 6144)
        self.assertEqual(self.config.hidden_act, "gelu")
        self.assertEqual(self.config.hidden_dropout_prob, 0.1)
        self.assertEqual(self.config.attention_probs_dropout_prob, 0.1)
        self.assertEqual(self.config.max_position_embeddings, 1024)

    def test_decoder(self):
        input_text = "今天天气不错"
        input_ids, _ = self.tokenizer.encode(input_text)

        out = sample_sequence(
            model=self.model_tf,
            model_decoder=self.model_decoder,
            context=input_ids,
            length=150,
            temperature=1.0,
            top_k=40,
            top_p=1.0,
            device=self.device,
        )

        text = self.tokenizer.decode(out)
        print(text)
