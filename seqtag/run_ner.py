#!/usr/bin/env python
# coding=utf-8

# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import pdb
import numpy as np
from tqdm import tqdm


from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import transformers
import evaluate

from ner_task_config import *


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class TaskArguments:
    """
    Arguments pertraining to which task to perfrom (e.g., trigger extraction, argument extraction) 
    """
    task_name: Optional[str] = field(
        default="trg", 
        metadata={"help": "one of {}".format(list(Task2LabelCol.keys()))}
        )
    pred_trg_file: Optional[str] = field(
        default=None, 
        metadata={"help": "trigger prediction file, if specified and task_name is 'arg', will do argument extraction based on predicted triggers rather than golden labeled triggers"}
    )
    pred_trg_tag_col: Optional[str] = field(
        default='pred_trg_tags', 
        metadata={"help": "column name of predicted triggers tags in pred_trg_file"}
    )
    output_filename: Optional[str] = field(
        default=None,
        metadata={"help": "you can specify the file name of model predictions"}
    )
    def __post_init__(self):
        self.task_name = self.task_name.lower()
        
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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((TaskArguments, ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        task_args, model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        task_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    task_args.just_infer = (not training_args.do_train and training_args.do_predict)
    # if task_args.just_infer:
    #     training_args.do_train, training_args.do_eval, training_args.do_predict = False, False, True

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # modified: loading and processing data
    logger.info("****** loading and processing data ******")
    raw_datasets = load_my_datasets_for_ner(data_args, task_args)

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        # features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["test"].column_names
        # features = raw_datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    else: 
        label_column_name = Task2LabelCol[task_args.task_name]
    

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    # labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    labels_are_int = False
    if labels_are_int:
        label_list = Task2Features[task_args.task_name][label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        # label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_list = Task2Features[task_args.task_name][label_column_name].feature.names
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Model has labels -> use them.
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
            # Reorganize `label_list` to match the ordering of the model.
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []

        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])

                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

            # set token_type_ids of trigger tokens to 1, used in argument extraction task
            if task_args.task_name == 'arg':
                trigger_word_ids = []
                for idx, tag in enumerate(examples['trg_tags'][i]):
                    if tag != 'O':
                        trigger_word_ids.append(idx)
                for idx, _ in enumerate(tokenized_inputs['token_type_ids'][i]):
                    if word_ids[idx] in trigger_word_ids:
                        tokenized_inputs['token_type_ids'][i][idx] = 1
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Metrics
    logger.info("*** loading metric ***")
    metric = evaluate.load('seqeval')


    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Initialize our Trainer
    logger.info("*** initializing trainer ***")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=None if (task_args.task_name=='arg' and task_args.pred_trg_file) or task_args.just_infer else compute_metrics
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        if task_args.task_name == 'trg':
            pred_df = predict_dataset.to_pandas()
            # pred_df['triggers'] = pred_df[['trg_tags', 'tokens']].apply(lambda x: tags2text(tags=x.trg_tags, tokens=x.tokens), axis=1)
            pred_df['triggers'] = pred_df[[f'event{i}_trigger' for i in range(1,7)]].apply(lambda row: [item for item in row if item!=None], axis=1)
            pred_df['pred_trg_tags'] = true_predictions
            pred_df['pred_triggers'] = pred_df[['pred_trg_tags', 'tokens']].apply(lambda x: tags2text(tags=x.pred_trg_tags, tokenized_src_text=x.tokens), axis=1)
            output_predictions_file = os.path.join(training_args.output_dir, task_args.output_filename) if task_args.output_filename else os.path.join(training_args.output_dir, "trg_predictions.csv")
            pred_df[['title_id', 'title', 'triggers', 'pred_triggers', 'trg_tags', 'pred_trg_tags']].to_csv(output_predictions_file, index=False)
        elif task_args.task_name == 'arg':
            pred_df = predict_dataset.to_pandas()
            pred_df['pred_arg_tags'] = true_predictions
            pred_df['pred_arguments'] = pred_df[['pred_arg_tags', 'tokens']].apply(lambda x: tags2text(tags=x.pred_arg_tags, tokenized_src_text=x.tokens), axis=1)
            if task_args.pred_trg_file: # if use predicted triggers
                pred_df['triggers'] = pred_df[['trg_tags', 'tokens']].apply(lambda x: tags2text(tags=x.trg_tags, tokenized_src_text=x.tokens), axis=1)
                pred_df['event_triples'] = pred_df[[f'event{i}_triple' for i in range(1,7)]].apply(lambda row: [list(item) for item in row if list(item)!=[]], axis=1)
                # pred_df['event_triples'] = pred_df['event_triples'].apply(list) 
                pred_df['pred_event_triples'] = pred_df.apply(combine_trg_args, axis=1)
                pred_df = pred_df.groupby(['title_id', 'title']).agg(list).sort_values(by='index', axis=0).reset_index()
                pred_df['event_triples'] = pred_df['event_triples'].apply(lambda x: x[0]) 
                # if not task_args.just_infer: pred_df['event_triples'] = pred_df.apply(agg_triples, axis=1)
                # if not task_args.just_infer: pred_df['event_triples'] = pred_df['event_triples'].apply(lambda x:x[0])
                output_predictions_file = os.path.join(training_args.output_dir, task_args.output_filename) if task_args.output_filename else os.path.join(training_args.output_dir, "pipeline_predictions.csv")
                pred_df[['title_id', 'title', 'event_triples', 'pred_event_triples']].to_csv(output_predictions_file, index=False)
            else: # if use golden triggers
                # pred_df['arguments'] = pred_df.apply(lambda x: tags2text(tags=x.arg_tags, tokens=x.tokens), axis=1)
                pred_df['triggers'] = pred_df[[f'event{i}_trigger' for i in range(1,7)]].apply(lambda row: [item for item in row if item!=None], axis=1)   
                pred_df = pred_df.groupby(['title_id', 'title']).agg(list).sort_values(by='index', axis=0).reset_index()
                pred_df['triggers'] = pred_df['triggers'].apply(lambda x: x[0])
                pred_df['pred_event_triples'] = ""
                for idx, row in tqdm(pred_df.iterrows(), total=len(pred_df)):
                    pred_triples = []
                    for i, arg in enumerate(row.pred_arguments):
                        trp = arg
                        trp.insert(1, row.triggers[i])
                        pred_triples.append(trp)
                    pred_df.at[idx, "pred_event_triples"] = pred_triples
                pred_df['pred_event_triples'] = pred_df['pred_event_triples'].apply(lambda x: [list(i) for i in x])
                pred_df['event_triples'] = pred_df['event_triples'].apply(lambda x: [i.tolist() for i in x]) 

                output_predictions_file = os.path.join(training_args.output_dir, task_args.output_filename) if task_args.output_filename else os.path.join(training_args.output_dir, "arg_predictions.csv")
                pred_df[['title_id', 'title', 'event_triples', 'pred_event_triples', 'arg_tags', 'pred_arg_tags']].to_csv(output_predictions_file, index=False)
        logger.info("save prediction output to %s" % output_predictions_file)



    kwargs = {"finetuned_from": Path(model_args.model_name_or_path).name, "tasks": "token-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()
