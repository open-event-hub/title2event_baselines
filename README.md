# Environment
```
pip install -r requirements.txt
```

# Scripts
`event_extraction/run_extraction.sh`: SegTag \
`event_extraction/question-answering/run_qa.sh`: SpanMRC and Seq2SeqMRC

## Take SeqTag as example:
Trigger Ex. training
```
CUDA_VISIBLE_DEVICES=3 python run_ner.py \
    --model_name_or_path bert-base-chinese \
    --task_name trg \
    --output_dir ./output/trg_ner/bert-base-chinese \
    --overwrite_output_dir \
    --train_file ../datasets/Title2Event/tagging_train.csv \
    --validation_file ../datasets/Title2Event/tagging_dev.csv \
    --test_file ../datasets/Title2Event/tagging_test.csv \
    --do_train  --do_eval  \
    --do_predict \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --num_train_epochs 30 \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --text_column_name tokens \
    --pred_trg_file ./output/trg_ner/bert-base-chinese/trg_predictions.csv \
```

Arg Ex. Training (with Golden Triggers)
```
CUDA_VISIBLE_DEVICES=3 python run_ner.py \
    --model_name_or_path bert-base-chinese \
    --task_name arg \
    --output_dir ./output/arg_ner/bert-base-chinese \
    --overwrite_output_dir \
    --train_file ../datasets/Title2Event/tagging_train.csv \
    --validation_file ../datasets/Title2Event/tagging_dev.csv \
    --test_file ../datasets/Title2Event/tagging_test.csv \
    --do_train  --do_eval  \
    --do_predict \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --num_train_epochs 30 \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --text_column_name tokens \
```

Arg Ex. Predict (Pipeline)
```
CUDA_VISIBLE_DEVICES=3 python run_ner.py \
    --model_name_or_path bert-base-chinese \
    --task_name trg \
    --output_dir ./output/trg_ner/bert-base-chinese \
    --overwrite_output_dir \
    --train_file ../datasets/Title2Event/tagging_train.csv \
    --validation_file ../datasets/Title2Event/tagging_dev.csv \
    --test_file ../datasets/Title2Event/tagging_test.csv \
    --do_train False --do_eval False \
    --do_predict \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --num_train_epochs 30 \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --text_column_name tokens \
    --pred_trg_file ./output/trg_ner/bert-base-chinese/trg_predictions.csv \
```