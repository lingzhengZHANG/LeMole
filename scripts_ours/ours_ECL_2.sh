export CUDA_VISIBLE_DEVICES=3

model_name=LinearLLM

for seq_len in 512 ;do
  for pred_len in 336;do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path /home/lzhang726/dataset/prediction/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_${seq_len}_${pred_len} \
      --model $model_name \
      --data custom \
      --features S \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --itr 1 \
      --t_dim 4\
      --batch_size 16\
      --is_revin 0

  done
done
