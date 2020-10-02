# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizer
)

from transformers.data.data_collator import DataCollator, DefaultDataCollator

from utils_couplet_trainer import CoupletDataset, Split

from bert_seq_model import BertForSeq2Seq

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model = BertForSeq2Seq.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        CoupletDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        CoupletDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )
        if training_args.do_eval
        else None
    )

    def predict(test_dataset: CoupletDataset, args: TrainingArguments, max_seq_length: int, model: BertForSeq2Seq, tokenizer: PreTrainedTokenizer):
        # Note that DistributedSampler samples randomly
        test_sampler = SequentialSampler(test_dataset) #if args.local_rank == -1 else DistributedSampler(eval_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, collate_fn=DefaultDataCollator().collate_batch)

        # Eval!
        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d", len(test_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        inputs_list = []
        preds_list = []
        labels_list = []
        model.eval()
        for batch in tqdm(test_dataloader, desc="Predicting"):
            for k, v in batch.items():
                batch[k] = v.to(args.device)

            with torch.no_grad():
                input_idx = torch.nonzero(batch["token_type_ids"])[0][1].item()
                input_ids = batch["input_ids"][:, :input_idx]
                token_type_ids = batch["token_type_ids"][:, :input_idx]
                output_ids = model.beam_search(input_ids=input_ids, token_type_ids=token_type_ids, sep_id=tokenizer.sep_token_id, beam_size=3, out_max_length=max_seq_length//2)
                output_tokens = tokenizer.decode(output_ids)
                preds_list.append(output_tokens)
                inputs_list.append(tokenizer.decode(input_ids[0][1:-1]))

                labels_list.append(tokenizer.decode(batch["input_ids"][batch["token_type_ids"]==1][:-1]))

        return inputs_list, labels_list, preds_list

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=None,
        prediction_loss_only=True
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     result = trainer.evaluate()

    #     output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    #     if trainer.is_world_master():
    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Eval results *****")
    #             for key, value in result.items():
    #                 logger.info("  %s = %s", key, value)
    #                 writer.write("%s = %s\n" % (key, value))

    #         results.update(result)

    # Predict
    if training_args.do_predict:
        test_dataset = CoupletDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )

        model = BertForSeq2Seq.from_pretrained(training_args.output_dir)
        model.to(training_args.device)
        inputs, labels, predictions = predict(test_dataset, training_args, data_args.max_seq_length, model, tokenizer)

        # predictions, label_ids, metrics = trainer.predict(test_dataset)
        # preds_list, _ = align_predictions(predictions, label_ids)

        # output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        # if trainer.is_world_master():
        #     with open(output_test_results_file, "w") as writer:
        #         for key, value in metrics.items():
        #             logger.info("  %s = %s", key, value)
        #             writer.write("%s = %s\n" % (key, value))

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_master():
            with open(output_test_predictions_file, "w") as writer:
                for input, label, pred in zip(inputs, labels, predictions):
                    line_txt = f'input: {input}\nlabel: {label}\npred:  {pred}\n'
                    writer.write(line_txt)
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
