from unittest import result
import datasets
from datasets import ClassLabel, Dataset, load_dataset, load_metric, Features
import pandas as pd
import pdb
from transformers import AutoTokenizer
dataset_tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
dataset_tokenizer.add_special_tokens({"additional_special_tokens": ['“', '”', "…", "……", "—", "——"]}) # unrecognized by tokenizer

Task2LabelCol = {'trg': "trg_tags", 'arg': "arg_tags", 'joint': ""}
trg_tags = ('O', 'B-T1', 'I-T1', 'B-T2', 'I-T2', 'B-T3', 'I-T3', 'B-T4', 'I-T4', 'B-T5', 'I-T5', 'B-T6', 'I-T6')
arg_tags = ('O', 'B-sbj', 'I-sbj', 'B-obj', 'I-obj')


Task2Features = {
    # trigger extraction
    "trg": Features(
            {
                # "id": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                Task2LabelCol['trg']: datasets.Sequence(
                    feature=datasets.features.ClassLabel(
                        names=sorted(list(trg_tags))
                    )
                )
            }
        ),
    # argument extraction
    "arg": Features(
            {
                # "id": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                Task2LabelCol['arg']: datasets.Sequence(
                    feature=datasets.features.ClassLabel(
                        names=sorted(list(arg_tags))
                    )
                )
            }
        )
}

def load_my_datasets_for_ner(data_args, task_args):
    if data_args.test_file is not None:
        tst_df = pd.read_csv(data_args.test_file)
        tst_df = preprocess(tst_df, task_args, is_test_data=True)
        test_dataset = Dataset.from_pandas(tst_df)
    # load dataframes
    if data_args.train_file is not None:
        trn_df = pd.read_csv(data_args.train_file)
        trn_df = preprocess(trn_df, task_args)
        train_dataset = Dataset.from_pandas(trn_df)
    if data_args.validation_file is not None:
        val_df = pd.read_csv(data_args.validation_file)
        val_df = preprocess(val_df, task_args)
        valid_dataset = Dataset.from_pandas(val_df)
    raw_dataset = {"train": train_dataset, 'validation': valid_dataset, 'test': test_dataset}
    return raw_dataset

def preprocess(df: pd.DataFrame, task_args, is_test_data=False):
    for col in df.columns:
        if col.endswith('tags') or col.endswith('triple') or col=='tokens':
            df[col] = df[col].fillna("None").apply(eval)
    if task_args.just_infer and is_test_data:
        df['tokens'] = df['title'].apply(lambda x: dataset_tokenizer.tokenize(x))
        df[Task2LabelCol[task_args.task_name]] = df['tokens'].apply(lambda x: ["O"]*len(x))

    if is_test_data and task_args.task_name == 'arg' and task_args.pred_trg_file: # do argument extraction based on predicted triggers rather than golden labeled triggers
        pred_trg_df = pd.read_csv(task_args.pred_trg_file)
        pred_trg_df[task_args.pred_trg_tag_col] = pred_trg_df[task_args.pred_trg_tag_col].apply(eval)
        df['trg_tags'] = pred_trg_df[task_args.pred_trg_tag_col]
        df = expand_arg_tags(df, ignore_arg_tags=True)

    elif task_args.task_name == 'arg':
        df = expand_arg_tags(df)
    return df

def expand_arg_tags(df: pd.DataFrame, ignore_arg_tags=False):
    """
    expand arg_tags for all events in a single sentence
    args:
        df: the loaded dataframe
        ignore_arg_tags: if set to True, will not align all golden labeled arguments to their corresponding triggers, 
            should be used when doing argument extraction based on predicted triggers rather than golden labeled triggers,
            since the number of predicted triggers may be less than total number of golden labeled triggers
    """
    df['trg_tags'] = df['trg_tags'].apply(split_trg_tags)
    if ignore_arg_tags:
        df = df.explode('trg_tags')
        # df['trg_tags'].fillna(value=None, inplace=True)
        df['arg_tags'] = df.tokens.apply(lambda x: ['O']*len(x))
        df = df[~(df['trg_tags'].isna())].reset_index()
    else:
        df['arg_tags'] = df[[f'event{i}_arg_tags' for i in range(1,7)]].values.tolist()
        df['event_triples'] = df[[f'event{i}_triple' for i in range(1,7)]].values.tolist()
        df['event_triples'] = df['event_triples'].apply(lambda x: [list(i) for i in x])
        df = df.explode(['trg_tags', 'arg_tags', 'event_triples']).reset_index()
        df = df[~(df['trg_tags'].isna()) & ~(df['arg_tags']!=None).isna()].reset_index()
    return df
    
def split_trg_tags(trg_tags):
    splited_tags = []
    if trg_tags:
        for i in range(1,7):
            tags = None
            if f'B-T{i}' in trg_tags or f'I-T{i}' in trg_tags:
                tags = [tag if str(i) in tag else "O" for tag in trg_tags]
            splited_tags.append(tags)
    if splited_tags == [None]*6: # if do argument extraction based on predicted triggers, and there's no predicted trigger for this instance, should at least keep one trg_tags so that this instance is not lost
        splited_tags[0] = trg_tags
    return splited_tags

def tags2text(tags, tokenized_src_text):
    """
    convert BIO tags into a list of tokens (only convert B and I which represent event elements, skip O) 
    args:
        tags: BIO tags of the example on token level
        tokenized_src_text: tokenized input text
    """
    assert len(tags)==len(tokenized_src_text), "tags and tokenized input text should be of same length"
    result = []
    cur_tkns = []
    cur_tags = []
    for i,tkn in enumerate(tokenized_src_text):
        if tags[i]=='O': continue
        if cur_tags==[] or tags[i].split('-')[1] == cur_tags[-1].split('-')[1]:
            cur_tags.append(tags[i])
            cur_tkns.append(tkn)
        else:
            result.append(dataset_tokenizer.convert_tokens_to_string(cur_tkns).replace(" ", ""))
            cur_tkns = [tkn]
            cur_tags = [tags[i]]
    if cur_tkns: result.append(dataset_tokenizer.convert_tokens_to_string(cur_tkns).replace(" ", ""))
    return result

def combine_trg_args(row):
    """
    combine triggers (golden labeled or predicted) with predicted arguments to form triplets
    """
    triple = row.pred_arguments
    if triple:
        if row.triggers:
            triple.insert(1, row.triggers[0])
        else:
            triple.insert(1, "")
        if len(triple) < 3: triple.append("")
    return triple

def agg_triples(row):
    """
    aggregate triplets belonging to the same source text
    """
    triples = []
    for i in range(1,7):
        if len(row[f'event{i}_triple'])>0:
            triples.append(list(row[f'event{i}_triple']))
    return triples