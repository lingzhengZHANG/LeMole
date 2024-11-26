export CUDA_VISIBLE_DEVICES=3
model_name=LinearLLM
for percent in 5 10; do
for w_o_dynamic_prompt in 0 1;do
for w_o_static_prompt in 0 1;do

for t_dim in 3 ;do
for seq_len in 1024;do
  for pred_len in 336  ;do #336
        python run.py \
        --data ETTm1\
        --model_id bs16_ETTm1_${seq_len}_${pred_len} \
        --model $model_name \
        --data_path ETTm1.csv\
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
        --itr 1\
        --percent $percent\
        --w_o_dynamic_prompt $w_o_dynamic_prompt\
        --w_o_static_prompt $w_o_static_prompt
    done
done
done

for t_dim in 4 ;do
for seq_len in 512;do
  for pred_len in 192  ;do #96
        python run.py \
        --data ETTm1\
        --model_id bs16_ETTm1_${seq_len}_${pred_len} \
        --model $model_name \
        --data_path ETTm1.csv\
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
        --itr 1\
        --percent $percent\
        --w_o_dynamic_prompt $w_o_dynamic_prompt\
        --w_o_static_prompt $w_o_static_prompt
    done
done
done

done
done
done
# for batch in 16; do
# for t_dim in 4;do
#     for pred_len in 192;do
#         python run.py \
#         --data ETTm1\
#         --model_id ETTm1_batch${batch}_${seq_len}_${pred_len} \
#         --model $model_name \
#         --data_path ETTm1.csv\
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
# for t_dim in 3;do
#     for pred_len in 336 720 ;do
#         python run.py \
#         --data ETTm1\
#         --model_id ETTm1_batch${batch}_1024_${pred_len} \
#         --model $model_name \
#         --data_path ETTm1.csv\
#         --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/\
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --features S \
#         --seq_len 1024 \
#         --pred_len $pred_len \
#         --batch_size $batch\
#         --t_dim $t_dim \
#         --is_revin 1\
#         --itr 3
#     done
# done
# done
# for batch in 16; do
# for t_dim in 3;do
#     for pred_len in  720 ;do
#         python run.py \
#         --data ETTm1\
#         --model_id ETTm1_batch${batch}_1024_${pred_len} \
#         --model $model_name \
#         --data_path ETTm1.csv\
#         --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/\
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --features S \
#         --seq_len 1024 \
#         --pred_len $pred_len \
#         --batch_size $batch\
#         --t_dim $t_dim \
#         --is_revin 1\
#         --itr 3
#     done
# done
# done
