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
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            activation=None,
            name="dense",
        )

        if self.num_labels == 1:
            self.loss_fct = tf.keras.losses.MeanSquaredError()
        else:
            self.loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

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
        self,
        inputs: Dict[str, tf.Tensor],
        training: Optional[bool] = False,
    ) -> Tuple[tf.Tensor, ...]:
        labels = inputs.get("label_id")
        bert_output = self.bert(inputs)

        pooled_output = bert_output["pooled_output"]
        pooled_output = self.dropout(pooled_output, training)
        logits = self.classifier(pooled_output)

        output_dict = {"logits": logits}

        if labels is not None:
            if self.num_labels == 1:
                loss = self.loss_fct(
                    tf.reshape(logits, (-1, )), tf.reshape(labels, (-1, )))
            else:
                loss = self.loss_fct(
                    tf.reshape(labels, (-1, )),
                    tf.reshape(logits, (-1, self.num_labels)))
            output_dict.update({"loss": loss})
            self.add_loss(loss)
            self.add_metric(self.acc_fct(labels, logits))

        return output_dict  # {"logits": logits, "loss": loss}


@TF_MODELS.register_module()
class TFBertForIQIYIDrama(TFPreTrainedModel):
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
        >>>     type="TFBertForIQIYIDrama",
        >>>     config=dict(
        >>>         type="ConfigBase",
        >>>         json_file="/workspace/models/nlp/chinese_wwm_ext/bert_config.json",
        >>>         num_labels=4,
        >>>     ),
        >>>     model_path="/workspace/models/nlp/chinese_wwm_ext/model_tf.bin",
        >>> )

    """

    def __init__(self, config: Dict, model_path: str = None) -> None:
        super().__init__(config)

        self.num_labels = self.config.num_labels
        self.bert = TFBertModel(self.config)
        self.dropout = tf.keras.layers.Dropout(self.config.hidden_dropout_prob)
        self.classifier1 = tf.keras.layers.Dense(
            self.num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            activation=None,
            name="dense1",
        )
        self.classifier2 = tf.keras.layers.Dense(
            self.num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            activation=None,
            name="dense2",
        )
        self.classifier3 = tf.keras.layers.Dense(
            self.num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            activation=None,
            name="dense3",
        )
        self.classifier4 = tf.keras.layers.Dense(
            self.num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            activation=None,
            name="dense4",
        )
        self.classifier5 = tf.keras.layers.Dense(
            self.num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            activation=None,
            name="dense5",
        )
        self.classifier6 = tf.keras.layers.Dense(
            self.num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02),
            activation=None,
            name="dense6",
        )

        self.label_indexs = tf.constant(
            [
                [0.8, 0.6, 0.4, 0.2],
                [0.6, 0.8, 0.6, 0.4],
                [0.4, 0.6, 0.8, 0.6],
                [0.2, 0.4, 0.6, 0.8],
            ],
            dtype=tf.float32,
        )

        self.loss_fct = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)

        self.acc_fct = tf.keras.metrics.RootMeanSquaredError()
        if model_path is not None:
            self._load_weights(model_path)

    def _load_weights(self, model_path: str = None) -> None:
        inputs = {
            "input_ids": tf.zeros((1, 1), dtype=tf.int32),
            "cht_indices": tf.zeros((1, 1), dtype=tf.int32),
        }
        self(inputs, training=False)
        self.load_weights(model_path)

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: Optional[bool] = False,
    ) -> Tuple[tf.Tensor, ...]:
        labels = inputs.get("labels")
        cht_indices = inputs.get("cht_indices")

        bert_output = self.bert(inputs)

        encoder_output = bert_output["encoder_output"]
        cht_output = tf.gather(
            encoder_output, cht_indices, axis=1, batch_dims=1)
        logits1 = tf.keras.activations.sigmoid(self.classifier1(cht_output))
        logits2 = tf.keras.activations.sigmoid(self.classifier2(cht_output))
        logits3 = tf.keras.activations.sigmoid(self.classifier3(cht_output))
        logits4 = tf.keras.activations.sigmoid(self.classifier4(cht_output))
        logits5 = tf.keras.activations.sigmoid(self.classifier5(cht_output))
        logits6 = tf.keras.activations.sigmoid(self.classifier6(cht_output))

        output_dict = {
            "logits1": logits1,
            "logits2": logits2,
            "logits3": logits3,
            "logits4": logits4,
            "logits5": logits5,
            "logits6": logits6,
        }

        if labels is not None:
            logits1 = tf.cast(logits1, tf.float32)
            logits2 = tf.cast(logits2, tf.float32)
            logits3 = tf.cast(logits3, tf.float32)
            logits4 = tf.cast(logits4, tf.float32)
            logits5 = tf.cast(logits5, tf.float32)
            logits6 = tf.cast(logits6, tf.float32)
            new_labels = tf.gather(self.label_indexs, labels, axis=0)
            loss1 = self.loss_fct(new_labels[:, 0],
                                  tf.reshape(logits1, (-1, self.num_labels)))
            loss2 = self.loss_fct(new_labels[:, 1],
                                  tf.reshape(logits2, (-1, self.num_labels)))
            loss3 = self.loss_fct(new_labels[:, 2],
                                  tf.reshape(logits3, (-1, self.num_labels)))
            loss4 = self.loss_fct(new_labels[:, 3],
                                  tf.reshape(logits4, (-1, self.num_labels)))
            loss5 = self.loss_fct(new_labels[:, 4],
                                  tf.reshape(logits5, (-1, self.num_labels)))
            loss6 = self.loss_fct(new_labels[:, 5],
                                  tf.reshape(logits6, (-1, self.num_labels)))
            # 可选，将label为0的loss变小10倍
            loss1 = tf.where(
                tf.reduce_all(
                    tf.equal(new_labels[:, 0], [0.8, 0.6, 0.4, 0.2]), axis=-1),
                loss1 * 0.1, loss1)
            loss2 = tf.where(
                tf.reduce_all(
                    tf.equal(new_labels[:, 1], [0.8, 0.6, 0.4, 0.2]), axis=-1),
                loss2 * 0.1, loss2)
            loss3 = tf.where(
                tf.reduce_all(
                    tf.equal(new_labels[:, 2], [0.8, 0.6, 0.4, 0.2]), axis=-1),
                loss3 * 0.1, loss3)
            loss4 = tf.where(
                tf.reduce_all(
                    tf.equal(new_labels[:, 3], [0.8, 0.6, 0.4, 0.2]), axis=-1),
                loss4 * 0.1, loss4)
            loss5 = tf.where(
                tf.reduce_all(
                    tf.equal(new_labels[:, 4], [0.8, 0.6, 0.4, 0.2]), axis=-1),
                loss5 * 0.1, loss5)
            loss6 = tf.where(
                tf.reduce_all(
                    tf.equal(new_labels[:, 5], [0.8, 0.6, 0.4, 0.2]), axis=-1),
                loss6 * 0.1, loss6)
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            output_dict.update({"loss": loss})
            self.add_loss(loss)
            rmse = self.acc_fct(
                labels,
                tf.concat(
                    [
                        tf.expand_dims(tf.argmax(logits1, axis=-1), 1),
                        tf.expand_dims(tf.argmax(logits2, axis=-1), 1),
                        tf.expand_dims(tf.argmax(logits3, axis=-1), 1),
                        tf.expand_dims(tf.argmax(logits4, axis=-1), 1),
                        tf.expand_dims(tf.argmax(logits5, axis=-1), 1),
                        tf.expand_dims(tf.argmax(logits6, axis=-1), 1),
                    ],
                    axis=1,
                ),
            )
            self.add_metric(1.0 / (1.0 + rmse), name="score")

        return output_dict  # {"logits": logits, "loss": loss}


@TF_MODELS.register_module()
class TFBertForIQIYIDramaRegress(TFPreTrainedModel):
    r"""TFBertForIQIYIDramaRegress for bert tensorflow

    Args:
        config (dict): config of bert model
        model_path (str): bert pretrained model file path

    Returns:
        dict[str, tf.Tensor]:
            - logits (tf.Tensor): (batch_size, num_labels)
            - loss (tf.Tensor): (batch_size, 1)

    Examples:

        >>> model_cfg = dict(
        >>>     type="TFBertForIQIYIDramaRegress",
        >>>     config=dict(
        >>>         type="ConfigBase",
        >>>         json_file="/workspace/models/nlp/chinese_wwm_ext/bert_config.json",
        >>>         num_labels=4,
        >>>     ),
        >>>     model_path="/workspace/models/nlp/chinese_wwm_ext/model_tf.bin",
        >>> )

    """

    def __init__(self, config: Dict, model_path: str = None) -> None:
        super().__init__(config)

        self.num_labels = self.config.num_labels
        self.bert = TFBertModel(self.config)
        self.dropout = tf.keras.layers.Dropout(self.config.hidden_dropout_prob)
        self.regresstion = tf.keras.layers.Dense(
            6,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                stddev=1.0e-5),
            activation=None,
            name="dense",
        )

        self.regress_labels = tf.constant([0.2, 0.4, 0.6, 0.8],
                                          dtype=tf.float32)

        # self.loss_fct \
        #   = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self.acc_fct = tf.keras.metrics.RootMeanSquaredError()
        if model_path is not None:
            self._load_weights(model_path)

    def _load_weights(self, model_path: str = None) -> None:
        inputs = {
            "input_ids": tf.zeros((1, 1), dtype=tf.int32),
            "cht_indices": tf.zeros((1, 1), dtype=tf.int32),
        }
        self(inputs, training=False)
        self.load_weights(model_path)

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: Optional[bool] = False,
    ) -> Tuple[tf.Tensor, ...]:
        labels = inputs.get("labels")
        cht_indices = inputs.get("cht_indices")

        bert_output = self.bert(inputs)

        encoder_output = bert_output["encoder_output"]
        cht_output = tf.gather(
            encoder_output, cht_indices, axis=1, batch_dims=1)
        logits = tf.keras.activations.relu(self.regresstion(cht_output))

        output_dict = {
            "logits": logits,
        }

        if labels is not None:
            # new_labels = labels * 0.2 + 0.2
            logits = tf.cast(logits, dtype=tf.float32)
            new_labels = tf.gather(self.regress_labels, labels, axis=0)
            # new_labels = tf.cast(new_labels, dtype=tf.float32)
            # loss = self.loss_fct(new_labels, logits)
            loss = tf.pow(tf.subtract(new_labels, logits), 2)

            label_count = tf.reduce_sum(
                tf.cast(tf.equal(new_labels, 0.2), dtype=tf.int32))
            all_size = tf.size(new_labels)
            factor = tf.subtract(
                1.0,
                tf.cast(tf.divide(label_count, all_size), dtype=tf.float32))
            loss = tf.where(tf.equal(new_labels, 0.2), loss * factor, loss)
            loss = tf.reduce_mean(loss)

            output_dict.update({"loss": loss})
            self.add_loss(loss)

            predictions = tf.argmin(
                # tf.abs(tf.expand_dims(logits, axis=-1) - self.regress_labels), axis=-1
                tf.abs(
                    tf.subtract(
                        tf.expand_dims(logits, axis=-1),
                        tf.cast(self.regress_labels, dtype=logits.dtype),
                    )),
                axis=-1,
            )

            rmse = self.acc_fct(labels, predictions)
            self.add_metric(1.0 / (1.0 + rmse), name="score")

        return output_dict  # {"logits": logits, "loss": loss}
