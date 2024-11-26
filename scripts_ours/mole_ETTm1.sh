export CUDA_VISIBLE_DEVICES=1
model_name=MoLE

for batch in 8; do
    for pred_len in 96 192 336 720 ;do
        python run.py \
        --data ETTm1\
        --model_id ETTm1_batch${batch}_${seq_len}_${pred_len} \
        --model $model_name \
        --data_path ETTm1.csv\
        --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/\
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --features M \
        --seq_len 512 \
        --pred_len $pred_len \
        --batch_size $batch\
        --is_revin 1
    done

done