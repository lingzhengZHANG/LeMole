export CUDA_VISIBLE_DEVICES=3

model_name=LinearLLM_F
for batch in 8 ; do
for t_dim in 5;do
    for pred_len in 96  336 720 ;do
        for seq_len in 1024 ;do 
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path /home/lzhang726/dataset/prediction/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_bs${batch}_${seq_len}_$pred_len \
      --model $model_name \
      --data custom \
      --features S \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --batch_size $batch\
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --t_dim $t_dim\
      --des 'Exp' \
      --is_revin 0\
      --itr 1
  done
done
done
done

for batch in 8 ; do
for t_dim in 5;do
    for pred_len in 192 ;do
        for seq_len in 512 ;do 
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path /home/lzhang726/dataset/prediction/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_bs${batch}_${seq_len}_$pred_len \
      --model $model_name \
      --data custom \
      --features S \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --batch_size $batch\
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --t_dim $t_dim\
      --des 'Exp' \
      --is_revin 0\
      --itr 1
  done
done
done
done