export CUDA_VISIBLE_DEVICES=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_quantum_retnet_classifier.py \
    --model_size quantum-retnet-small_classifier \
    --output_dir checkpoints/quantum_small_classifier \
    --do_train --do_eval \
    --prediction_loss_only \
    --remove_unused_columns False \
    --learning_rate 6e-4 \
    --weight_decay 0.01 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --max_steps 3 \
    --logging_steps 1 \
    --eval_steps 1000 \
    --save_steps 3 \
    --save_total_limit 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --metric_for_best_model eval_loss \
    --greater_is_better False 
#--fp16 True \ --load_best_model_at_end True \
