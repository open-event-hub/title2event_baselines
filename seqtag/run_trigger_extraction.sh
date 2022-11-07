# run trigger extraction using sequence-tagging model
MODEL=bert-base-chinese
OUTPUT_DIR=./output/trg_seqtag/$MODEL

python run_ner.py \
    --model_name_or_path $MODEL \
    --task_name trg \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --train_file ../dataset/tagging/tagging_train.csv \
    --validation_file ../dataset/tagging/tagging_dev.csv \
    --test_file ../dataset/tagging/tagging_test.csv \
    --do_train \
    --do_eval  \
    --do_predict \
    --output_filename trg_predictions.csv \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --num_train_epochs 30 \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --text_column_name tokens 
