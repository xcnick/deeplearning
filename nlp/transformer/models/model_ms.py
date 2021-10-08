from typing import Tuple, Optional, Dict
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore import load_checkpoint, load_param_into_net

from transformer.transformer.bert_ms import MSBertModel
from transformer.transformer.utils_ms import MSPreTrainedModel
from transformer.builder import MS_MODELS


@MS_MODELS.register_module()
class MSBertForSequenceClassification(MSPreTrainedModel):
    r"""SequenceClassification for bert mindspore

    Args:
        config (dict): config of bert model
        model_path (str): bert pretrained model file path

    Returns:
        dict[str, ms.Tensor]:
            - logits (ms.Tensor): (batch_size, num_labels)
            - loss (ms.Tensor): (batch_size, 1)

    Examples:

        >>> model_cfg = dict(
        >>>     type="MSBertForSequenceClassification",
        >>>     config=dict(
        >>>         type="ConfigBase",
        >>>         json_file="/workspace/models/nlp/chinese_wwm_ext/bert_config.json",
        >>>         num_labels=14,
        >>>     ),
        >>>     model_path="/workspace/models/nlp/chinese_wwm_ext/model_tf.bin",
        >>> )

    """

    def __init__(self, config: Dict, model_path: str = None) -> None:
        super().__init__(config)

        self.num_labels = self.config.num_labels
        self.bert = MSBertModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(
            self.config.hidden_size,
            self.num_labels,
            weight_init=TruncatedNormal(self.config.initializer_range),
        )

        if self.num_labels == 1:
            self.loss_fct = nn.MSELoss()
        else:
            self.loss_fct = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

        # self.acc_fct = tf.keras.metrics.SparseCategoricalAccuracy()
        if model_path is not None:
            self._load_weights(model_path)

    def _load_weights(self, model_path: str = None) -> None:
        param_dict = load_checkpoint(model_path)
        load_param_into_net(self, param_dict)

    def construct(
        self, input_ids: ms.Tensor, attention_mask: ms.Tensor, label_id: ms.Tensor
    ) -> Tuple[ms.Tensor, ...]:
        labels = label_id
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = logits

        if labels is not None:
            if self.num_labels == 1:
                loss = self.loss_fct(logits.reshape((-1,)), labels.reshape((-1,)))
            else:
                loss = self.loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1,)))
            outputs = outputs + (loss,)
            # self.add_loss(loss)
            # self.add_metric(self.acc_fct(labels, logits))

        return outputs  # {"logits": logits, "loss": loss}
