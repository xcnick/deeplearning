import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from transformers import BertPreTrainedModel, BertModel


class BertForSeq2Seq(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.decoder = LMPrediction(config, self.bert.embeddings.word_embeddings.weight)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, calc_loss=True):
        input_shape = input_ids.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        ones = torch.ones((1, 1, seq_len, seq_len), device=input_ids.device)
        a_mask = ones.tril()  # 下三角矩阵
        s_ex12 = token_type_ids.unsqueeze(1).unsqueeze(2)
        s_ex13 = token_type_ids.unsqueeze(1).unsqueeze(3)
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask

        seq_output = self.bert(
            input_ids=input_ids, attention_mask=a_mask, token_type_ids=token_type_ids
        )[0]

        prediction_scores = self.decoder(seq_output)

        outputs = (prediction_scores,)

        if calc_loss:
            prediction_scores = prediction_scores[:, :-1].contiguous()
            labels = input_ids[:, 1:].contiguous()
            target_mask = token_type_ids[:, 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            active_loss = target_mask == 1
            loss = loss_fct(prediction_scores[active_loss], labels[active_loss])

            outputs = (loss,) + outputs

        return outputs

    def generate(self, input_ids=None, token_type_ids=None, sep_id=102):
        input_len = input_ids.shape[1] - 2

        for step in range(input_len):
            scores = self.forward(
                input_ids=input_ids, token_type_ids=token_type_ids, calc_loss=False
            )[0]
            next_chars = torch.argmax(scores, dim=-1)[:, -1].unsqueeze(0)
            next_token_type_ids = torch.ones_like(next_chars, device=input_ids.device)
            input_ids = torch.cat((input_ids, next_chars), dim=1)
            token_type_ids = torch.cat((token_type_ids, next_token_type_ids), dim=1)

        return input_ids[0]


    def beam_search(
        self, input_ids=None, token_type_ids=None, sep_id=102, beam_size=1, out_max_length=32
    ):
        """
        beam-search操作
        """
        # sep_id = word2ix["[SEP]"]
        # 用来保存输出序列
        output_ids = [[]]
        # 用来保存累计得分
        output_scores = torch.zeros(input_ids.shape[0], device=input_ids.device)
        for step in range(out_max_length):

            scores = self.forward(
                input_ids=input_ids, token_type_ids=token_type_ids, calc_loss=False
            )[0]
            # print(scores.shape)
            if step == 0:
                # 重复beam-size次 输入ids
                input_ids = input_ids.view(1, -1).repeat(beam_size, 1)
                token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
            ## 计算log 分值 (beam_size, vocab_size)
            logit_score = torch.log_softmax(scores, dim=-1)[:, -1]
            logit_score = output_scores.view(-1, 1) + logit_score  # 累计得分
            ## 取topk的时候我们是展平了然后再去调用topk函数
            # 展平
            logit_score = logit_score.view(-1)
            hype_score, hype_pos = torch.topk(logit_score, beam_size)
            indice1 = hype_pos // scores.shape[-1]  # 行索引
            indice2 = hype_pos % scores.shape[-1]  # 列索引

            # 下面需要更新一下输出了
            new_hype_scores = []
            new_hype_ids = []
            # 为啥有这个[],就是因为要过滤掉结束的序列。
            next_chars = []  # 用来保存新预测出来的一个字符，继续接到输入序列后面，再去预测新字符
            for i_1, i_2, score in zip(indice1, indice2, hype_score):
                i_1 = i_1.item()
                i_2 = i_2.item()
                score = score.item()

                hype_id = output_ids[i_1] + [i_2]  # 保存所有输出的序列，而不仅仅是新预测的单个字符

                if i_2 == sep_id:
                    # 说明解码到最后了
                    if score == torch.max(hype_score).item():
                        # 说明找到得分最大的那个序列了 直接返回即可
                        return hype_id[:-1]
                    else:
                        # 完成一个解码了，但这个解码得分并不是最高，因此的话需要跳过这个序列
                        beam_size -= 1
                else:
                    new_hype_ids.append(hype_id)
                    new_hype_scores.append(score)
                    next_chars.append(i_2)  # 收集一下，需要连接到当前的输入序列之后

            output_ids = new_hype_ids

            output_scores = torch.tensor(
                new_hype_scores, dtype=torch.float32, device=input_ids.device
            )
            # 现在需要重新构造输入数据了，用上一次输入连接上这次新输出的字符，再输入bert中预测新字符
            input_ids = input_ids[: len(output_ids)].contiguous()  # 截取，因为要过滤掉已经完成预测的序列
            token_type_ids = token_type_ids[: len(output_ids)].contiguous()

            next_chars = torch.tensor(next_chars, dtype=torch.long, device=input_ids.device).view(
                -1, 1
            )
            next_token_type_ids = torch.ones_like(next_chars, device=input_ids.device)
            # 连接
            input_ids = torch.cat((input_ids, next_chars), dim=1)
            token_type_ids = torch.cat((token_type_ids, next_token_type_ids), dim=1)
            if beam_size < 1:
                break

        # 如果达到最大长度的话 直接把得分最高的输出序列返回把
        return output_ids[output_scores.argmax().item()]


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu
    # gelu_new = torch.jit.script(gelu_new)


class LMPrediction(nn.Module):
    def __init__(self, config, embedding_weight):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.decoder.weight = embedding_weight

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        return hidden_states
