export CUDA_VISIBLE_DEVICES=3
model_name=MoLE

for batch in 8; do
    for pred_len in 96 192 336 720 ;do
        python run.py \
        --data custom\
        --model_id traffic_batch${batch}_${seq_len}_${pred_len} \
        --model $model_name \
        --data_path traffic.csv\
        --root_path /mnt/users/lzhang726/dataset/prediction/traffic/\
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --features M \
        --seq_len 512 \
        --pred_len $pred_len \
        --batch_size $batch\
        --is_revin 1
    done
done