# first, lets train argument extraction SpanMRC model on golden triggers
python run_qa.py \
    --model_name_or_path bert-base-chinese \
    --output_dir ./output/arg_spanmrc/bert-base-chinese \
    --overwrite_output_dir \
    --version_2_with_negative \
    --train_file ../dataset/train.csv \
    --validation_file ../dataset/dev.csv \
    --test_file ../dataset/test.csv \
    --do_train \
    --do_eval \
    --do_predict \
    --output_filename arg_predictions.csv \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --num_train_epochs 30 \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 

# then, given the extraced triggers, we could use this model to predict arguments in a pipeline manner
PRED_TRIGGER_FILE=../seqtag/output/trg_seqtag/bert-base-chinese/trg_predictions.csv
python run_qa.py \
    --model_name_or_path ./output/arg_spanmrc/bert-base-chinese \
    --output_dir ./output/arg_spanmrc/bert-base-chinese \
    --overwrite_output_dir \
    --version_2_with_negative \
    --train_file ../dataset/train.csv \
    --validation_file ../dataset/dev.csv \
    --test_file ../dataset/test.csv \
    --pred_trg_file  $PRED_TRIGGER_FILE \
    --do_train false \
    --do_eval false \
    --do_predict \
    --output_filename pipeline_predictions.csv \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --num_train_epochs 30 \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 