export CUDA_VISIBLE_DEVICES=0

python train.py \
    --model_size 300m \
    --output_dir checkpoints \
    --do_train --do_eval \
    --prediction_loss_only \
    --remove_unused_columns False \
    --learning_rate 6e-4 \
    --weight_decay 0.01 \
    --max_steps 30 \
    --logging_steps 10 \
    --eval_steps 10 \
    --save_steps 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16