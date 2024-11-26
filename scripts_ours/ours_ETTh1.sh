export CUDA_VISIBLE_DEVICES=0
model_name=LinearLLM
seq_len=512
for t_dim in 4;do
for seq_len in 512 ;do
  for pred_len in  96 ;do
        python run.py \
        --data ETTh1\
        --model_id bs16_ETTh1_${seq_len}_${pred_len}_${t_dim} \
        --model $model_name \
        --data_path ETTh1.csv\
        --root_path /home/lzhang726/dataset/prediction/ETT-small/\
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --features S \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --batch_size 16\
        --t_dim $t_dim \
        --is_revin 1\
        --itr 1
    done
done
done

# for batch in 16; do
# for t_dim in 5;do
#     for pred_len in 192;do
#         python run.py \
#         --data ETTh1\
#         --model_id ETTh1_batch${batch}_${seq_len}_${pred_len} \
#         --model $model_name \
#         --data_path ETTh1.csv\
#         --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/\
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --features S \
#         --seq_len 512 \
#         --pred_len $pred_len \
#         --batch_size $batch\
#         --t_dim $t_dim \
#         --is_revin 1\
#         --itr 3
#     done
# done
# done
# for batch in 16; do
# for t_dim in 5;do
#     for pred_len in 336 720 ;do
#         python run.py \
#         --data ETTh1\
#         --model_id ETTh1_batch${batch}_${seq_len}_${pred_len} \
#         --model $model_name \
#         --data_path ETTh1.csv\
#         --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/\
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --features S \
#         --seq_len 512 \
#         --pred_len $pred_len \
#         --batch_size $batch\
#         --t_dim $t_dim \
#         --is_revin 1\
#         --itr 3
#     done
# done
# done
# for batch in 16; do
# for t_dim in 1;do
#     for pred_len in  720 ;do
#         python run.py \
#         --data ETTh1\
#         --model_id ETTh1_batch${batch}_${seq_len}_${pred_len} \
#         --model $model_name \
#         --data_path ETTh1.csv\
#         --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/\
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --features S \
#         --seq_len 512 \
#         --pred_len $pred_len \
#         --batch_size $batch\
#         --t_dim $t_dim \
#         --is_revin 1\
#         --itr 3
#     done
# done
# done
