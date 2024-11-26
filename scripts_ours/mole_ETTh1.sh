export CUDA_VISIBLE_DEVICES=0
model_name=MoLE

for batch in 8; do
    for pred_len in 96 192 336 720 ;do
        python run.py \
        --data ETTh1\
        --model_id ETTh1_batch${batch}_${seq_len}_${pred_len} \
        --model $model_name \
        --data_path ETTh1.csv\
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
