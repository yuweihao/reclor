export RECLOR_DIR=reclor_data
export TASK_NAME=reclor
export MODEL_NAME=roberta-large

CUDA_VISIBLE_DEVICES=0 python run_multiple_choice.py \
    --model_type roberta \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $RECLOR_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 1   \
    --per_gpu_train_batch_size 1   \
    --gradient_accumulation_steps 24 \
    --learning_rate 1e-05 \
    --num_train_epochs 10.0 \
    --output_dir Checkpoints/$TASK_NAME/${MODEL_NAME} \
    --fp16 \
    --logging_steps 200 \
    --save_steps 200 \
    --adam_betas "(0.9, 0.98)" \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --weight_decay 0.01