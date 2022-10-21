from bdb import set_trace
from cProfile import label
import pdb
import pandas as pd
from datasets import Dataset

def load_my_datasets_for_mrc(data_args, task_args):
    raw_dataset = {}
    if data_args.test_file is not None:
        tst_df = pd.read_csv(data_args.test_file).fillna("None")
        tst_df = preprocess(tst_df, task_args, is_pred_data=True)
        test_dataset = Dataset.from_pandas(tst_df)
        raw_dataset['test'] = test_dataset
    if data_args.train_file is not None:
        trn_df = pd.read_csv(data_args.train_file).fillna("None")
        trn_df = preprocess(trn_df, task_args)
        train_dataset = Dataset.from_pandas(trn_df)
        raw_dataset['train'] = train_dataset
    if data_args.validation_file is not None:
        val_df = pd.read_csv(data_args.validation_file).fillna("None")
        val_df = val_df
        val_df = preprocess(val_df, task_args)
        valid_dataset = Dataset.from_pandas(val_df)
        raw_dataset['validation'] = valid_dataset
    # raw_dataset = {"train": train_dataset, 'validation': valid_dataset, 'test': test_dataset}
    # pdb.set_trace()
    return raw_dataset

def preprocess(df, task_arg, is_pred_data=False):
    def find_answers(row):
        sbj_ans, obj_ans = {"text": [], "answer_start": []}, {"text": [], "answer_start": []}
        if task_arg.is_extractive:
            if row.triple and row.triple[0] and row.triple[0] in row.title: 
                sbj_ans['text'] = [row.triple[0]]
                sbj_ans['answer_start'] = [row.title.index(row.triple[0])]
            if row.triple and row.triple[2] and row.triple[2] in row.title: 
                obj_ans['text'] = [row.triple[2]]
                obj_ans['answer_start'] = [row.title.index(row.triple[2])]
        else:
            if row.triple and row.triple[0]: sbj_ans['text'] = [row.triple[0]]
            if row.triple and row.triple[2]: obj_ans['text'] = [row.triple[2]]
        return [sbj_ans, obj_ans]
    for col in df.columns:
        if col.endswith('triple'):
            df[col] = df[col].apply(eval)
    # pdb.set_trace()
    df['triple'] = df[[f'event{i}_triple' for i in range(1,7)]].apply(lambda row: [x for x in row], axis=1)
    df['gold_answer_triples'] = df['triple'].apply(lambda x: x if None not in x else x[:x.index(None)])
    if is_pred_data and task_arg.pred_trg_file:
        
        pred_trg_df = pd.read_csv(task_arg.pred_trg_file)
        df['trigger'] = pred_trg_df[task_arg.pred_trg_col].apply(eval)
        df['trigger'] = df['trigger'].apply(lambda x: (x + [None] * (6-len(x)))[:6])
        df['triple'] = df[[f'event{i}_triple' for i in range(1,7)]].apply(lambda row: [x for x in row if x !=None], axis=1)
        df['triple'] = df['triple'].apply(lambda x: (x + [[]] * (6-len(x)))[:6])

    else: 
        df['trigger'] = df['triple'].apply(lambda x: [tu[1] if tu else None for tu in x])
    # pdb.set_trace()
    df['id'] = df['triple'].apply(lambda x: [str(i) for i, _ in enumerate(x)])
    df = df.explode(['triple', 'id', 'trigger'])[['id', 'title_id', 'title', 'trigger', 'triple', 'gold_answer_triples']].dropna().reset_index(drop=True)
    df['title_id'] = df['title_id'].apply(lambda x: str(x))
    df['id'] = df['title_id'] + "-" + df['id']
        
    df['question'] = df['trigger'].apply(lambda x: [f"动作{x}的主体是？", f"动作{x}的客体是？"])
    df['q_id'] = [["sbj", "obj"]]*len(df)
    df['answers'] = df.apply(find_answers, axis=1)
    df.rename({"title": "context"}, axis=1, inplace=True)
    df = df.explode(['question', 'answers', 'q_id']).reset_index(drop=True)
    df['id'] = df['id'] + "_" + df['q_id']
    df = df.drop(labels='q_id', axis=1)
    return df