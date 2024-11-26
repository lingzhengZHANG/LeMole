export CUDA_VISIBLE_DEVICES=2

model_name=MoLE

for seq_len in 512 ;do
  for pred_len in 96 192 336 720;do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path /mnt/users/lzhang726/dataset/prediction/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_${seq_len}_${pred_len} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 8\
      --is_revin 1

  done
done
