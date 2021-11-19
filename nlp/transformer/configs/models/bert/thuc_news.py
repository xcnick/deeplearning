_base_ = ["../../datasets/THUCNews_tfrecord.py", "../../runtime/tf_runtime.py"]

model = dict(
    type="TFBertForSequenceClassification",
    config=dict(
        type="ConfigBase",
        json_file="/workspace/models/nlp/chinese_wwm_ext/bert_config.json",
        num_labels=14,
    ),
    model_path="/workspace/models/nlp/chinese_wwm_ext/model_tf.bin")
