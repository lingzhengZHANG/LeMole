export CUDA_VISIBLE_DEVICES=1

model_name=LinearLLM
for t_dim in 4;do
for seq_len in  512;do
  for pred_len in 336  ;do #720
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path /home/lzhang726/dataset/prediction/electricity/ \
      --data_path electricity.csv \
      --model_id bs16_ECL_${seq_len}_$pred_len \
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
      --batch_size 16\
      --t_dim $t_dim\
      --des 'Exp' \
      --is_revin 0\
      --itr 1
  done
done
done

# for t_dim in 3 ;do
# for seq_len in 1024;do
#   for pred_len in 192;do
#     python -u run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path /mnt/users/lzhang726/dataset/prediction/electricity/ \
#       --data_path electricity.csv \
#       --model_id ECL_${seq_len}_$pred_len \
#       --model $model_name \
#       --data custom \
#       --features S \
#       --seq_len $seq_len \
#       --label_len 48 \
#       --pred_len $pred_len \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 1 \
#       --dec_in 1 \
#       --c_out 1 \
#       --t_dim $t_dim\
#       --des 'Exp' \
#       --is_revin 0\
#       --itr 3
#   done
# done
# done


# for t_dim in 4 ;do
# for seq_len in 512;do
#   for pred_len in 336 ;do
#     python -u run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path /mnt/users/lzhang726/dataset/prediction/electricity/ \
#       --data_path electricity.csv \
#       --model_id ECL_${seq_len}_$pred_len \
#       --model $model_name \
#       --data custom \
#       --features S \
#       --seq_len $seq_len \
#       --label_len 48 \
#       --pred_len $pred_len \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 1 \
#       --dec_in 1 \
#       --c_out 1 \
#       --t_dim $t_dim\
#       --des 'Exp' \
#       --is_revin 0\
#       --itr 3
#   done
# done
# done

# for t_dim in 2 ;do
# for seq_len in 512;do
#   for pred_len in 720;do
#     python -u run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path /mnt/users/lzhang726/dataset/prediction/electricity/ \
#       --data_path electricity.csv \
#       --model_id ECL_${seq_len}_$pred_len \
#       --model $model_name \
#       --data custom \
#       --features S \
#       --seq_len $seq_len \
#       --label_len 48 \
#       --pred_len $pred_len \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 1 \
#       --dec_in 1 \
#       --c_out 1 \
#       --t_dim $t_dim\
#       --des 'Exp' \
#       --is_revin 0\
#       --itr 3
#   done
# done
# done


