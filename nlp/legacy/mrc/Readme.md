# 阅读理解算法

## SQuAD数据集

- 格式

```
{
    "data": [
        {
            "title": "Super_Bowl_50",
            "paragraphs": [
                {
                    "context": " numerals 50.",
                    "qas": [
                        {
                            "answers": [
                                {
                                    "answer_start": 177,
                                    "text": "Denver Broncos"
                                },
                                {
                                    "answer_start": 177,
                                    "text": "Denver Broncos"
                                },
                                {
                                    "answer_start": 177,
                                    "text": "Denver Broncos"
                                }
                            ],
                            "question": "Which NFL team represented the AFC at Super Bowl 50?",
                            "id": "56be4db0acb8001400a502ec"
                        }

                    ]
                }
            ]
        }
    ],
    "version": "1.1"
}
```

- 评价标准：
  - F1: 将pred的短语切成词，与target计算F1；
  - EM: Exact Match完全匹配，pred与target完全一致才算正确。


# 训练

```
python -m torch.distributed.launch --nproc_per_node=1 run_squad.py   --model_type bert   --model_name_or_path /home/orient/chi/model/chinese_wwm_ext_1   --do_train   --do_eval   --train_file /home/orient/chi/data/cmrc2018/squad-style-data/cmrc2018_train.json   --predict_file /home/orient/chi/data/cmrc2018/squad-style-data/cmrc2018_dev.json   --per_gpu_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 2.0   --max_seq_length 384   --doc_stride 128   --output_dir /home/orient/chi/data/out/cmrc2018/ --overwrite_output_dir --do_lower_case --save_steps=10000
```