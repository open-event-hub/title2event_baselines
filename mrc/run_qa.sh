
CUDA_VISIBLE_DEVICES=5 python run_seq2seq_qa.py \
    --model_name_or_path ../output/arg_seq2seq_qa/mt5-base \
    --output_dir ../output/arg_seq2seq_qa/mt5-base \
    --overwrite_output_dir \
    --version_2_with_negative \
    --train_file ../../datasets/Title2Event/train.csv \
    --validation_file ../../datasets/Title2Event/dev.csv \
    --test_file ../../datasets/Title2Event/test.csv \
    --eval_accumulation_steps 1 \
    --predict_with_generate \
    --version_2_with_negative \
    --do_train False --do_eval False \
    --learning_rate 1e-4 \
    --do_predict \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 30 \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 \
    --pred_trg_file ../output/trg_ner/bert-base-chinese/trg_predictions.csv \

# CUDA_VISIBLE_DEVICES=2 python run_qa.py \
#     --model_name_or_path ../output/arg_qa/bert-base-chinese \
#     --output_dir ../output/arg_qa/bert-base-chinese \
#     --overwrite_output_dir \
#     --version_2_with_negative \
#     --train_file ../../datasets/Title2Event/train.csv \
#     --validation_file ../../datasets/Title2Event/dev.csv \
#     --test_file ../../datasets/Title2Event/test.csv \
#     --do_train  False --do_eval False \
#     --do_predict \
#     --per_device_train_batch_size=32 \
#     --per_device_eval_batch_size=32 \
#     --num_train_epochs 30 \
#     --save_strategy epoch \
#     --logging_strategy epoch \
#     --evaluation_strategy epoch \
#     --save_total_limit 1 \
#     --load_best_model_at_end \
#     --metric_for_best_model eval_f1 \
#     --pred_trg_file ../output/trg_ner/bert-base-chinese/trg_predictions.csv \


# CUDA_VISIBLE_DEVICES=7 python run_seq2seq_qa.py \
#   --model_name_or_path t5-base \
#   --dataset_name squad_v2 \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ../output/tmp/t5-base