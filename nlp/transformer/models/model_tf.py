from typing import Tuple, Optional, Dict
import tensorflow as tf

from transformer.transformer.bert_tf import TFBertModel
from transformer.transformer.utils_tf import TFPreTrainedModel
from transformer.builder import TF_MODELS


@TF_MODELS.register_module()
class TFBertForSequenceClassification(TFPreTrainedModel):
    r"""SequenceClassification for bert tensorflow

    Args:
        config (dict): config of bert model
        model_path (str): bert pretrained model file path

    Returns:
        dict[str, tf.Tensor]:
            - logits (tf.Tensor): (batch_size, num_labels)
            - loss (tf.Tensor): (batch_size, 1)

    Examples:

        >>> model_cfg = dict(
        >>>     type="TFBertForSequenceClassification",
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
        self.bert = TFBertModel(self.config)
        self.dropout = tf.keras.layers.Dropout(self.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            self.num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            activation=None,
            name="dense",
        )

        if self.num_labels == 1:
            self.loss_fct = tf.keras.losses.MeanSquaredError()
        else:
            self.loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.acc_fct = tf.keras.metrics.SparseCategoricalAccuracy()
        if model_path is not None:
            self._load_weights(model_path)

    def _load_weights(self, model_path: str = None) -> None:
        inputs = {
            "input_ids": tf.zeros((1, 1), dtype=tf.int32),
        }
        self(inputs, training=False)
        self.load_weights(model_path)

    def call(
        self, inputs: Dict[str, tf.Tensor], training: Optional[bool] = False,
    ) -> Tuple[tf.Tensor, ...]:
        labels = inputs.get("label_id")
        bert_output = self.bert(inputs)

        pooled_output = bert_output["pooled_output"]
        pooled_output = self.dropout(pooled_output, training)
        logits = self.classifier(pooled_output)

        output_dict = {"logits": logits}

        if labels is not None:
            if self.num_labels == 1:
                loss = self.loss_fct(tf.reshape(logits, (-1,)), tf.reshape(labels, (-1,)))
            else:
                loss = self.loss_fct(
                    tf.reshape(labels, (-1,)), tf.reshape(logits, (-1, self.num_labels))
                )
            output_dict.update({"loss": loss})
            self.add_loss(loss)
            self.add_metric(self.acc_fct(labels, logits))

        return output_dict  # {"logits": logits, "loss": loss}
