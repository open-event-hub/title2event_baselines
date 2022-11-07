# first, lets train argument extraction Seq2SeqMRC model on golden triggers
python run_seq2seq_qa.py \
    --model_name_or_path google/mt5-base \
    --output_dir ./output/arg_seq2seqmrc/mt5-base \
    --overwrite_output_dir \
    --version_2_with_negative \
    --train_file ../dataset/train.csv \
    --validation_file ../dataset/dev.csv \
    --test_file ../dataset/test.csv \
    --output_filename arg_predictions.csv \
    --eval_accumulation_steps 1 \
    --predict_with_generate \
    --do_train \
    --do_eval \
    --do_predict \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 30 \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 
  
# then, given the extraced triggers, we could use this model to predict arguments in a pipeline manner
PRED_TRIGGER_FILE=../seqtag/output/trg_seqtag/bert-base-chinese/trg_predictions.csv
python run_seq2seq_qa.py \
    --model_name_or_path ./output/arg_seq2seqmrc/mt5-base \
    --output_dir ./output/arg_seq2seqmrc/mt5-base \
    --overwrite_output_dir \
    --version_2_with_negative \
    --train_file ../dataset/train.csv \
    --validation_file ../dataset/dev.csv \
    --test_file ../dataset/test.csv \
    --pred_trg_file  $PRED_TRIGGER_FILE \
    --output_filename pipeline_predictions.csv \
    --eval_accumulation_steps 1 \
    --predict_with_generate \
    --do_train false \
    --do_eval false \
    --do_predict \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 30 \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 