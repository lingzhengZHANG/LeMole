export CUDA_VISIBLE_DEVICES=3

model_name=FITS

for H_order in 10
do
for seq_len in 720
do
# for m in 1
# do
# for seed in 514 1919 810 0 #114
# do 
for bs in 64 #256 #32 64 # 128 256
do

# python -u run.py \
#   --is_training 1 \
#   --root_path /mnt/users/lzhang726/dataset/prediction/traffic/ \
#   --data_path traffic.csv  \
#   --model_id traffic_$seq_len'_'96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --enc_in 862 \
#   --des 'Exp' \
#   --H_order $H_order \
#   --gpu 0 \
#   --patience 20 \
#   --train_epochs 50\
#   --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/custom_abl/$m'_'$model_name'_'custom_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

# #   echo "Done $model_name'_'custom_$seq_len'_'96'_H'$H_order'_s'$seed"

# python -u run.py \
#   --is_training 1 \
#   --root_path /mnt/users/lzhang726/dataset/prediction/traffic/ \
#   --data_path traffic.csv  \
#   --model_id traffic_$seq_len'_'192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 192 \
#   --enc_in 862 \
#   --des 'Exp' \
#   --H_order $H_order \
#   --gpu 0 \
#   --patience 20 \
#   --train_epochs 50\
#   --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/custom_abl/$m'_'$model_name'_'custom_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

# #   echo "Done $model_name'_'custom_$seq_len'_'192'_H'$H_order'_s'$seed"

# python -u run.py \
#   --is_training 1 \
#   --root_path /mnt/users/lzhang726/dataset/prediction/traffic/ \
#   --data_path traffic.csv  \
#   --model_id traffic_$seq_len'_'336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 336 \
#   --enc_in 862 \
#   --des 'Exp' \
#   --H_order $H_order \
#   --gpu 0 \
#   --patience 20 \
#   --train_epochs 50\
#   --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/custom_abl/$m'_'$model_name'_'custom_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

#   echo "Done $model_name'_'custom_$seq_len'_'336'_H'$H_order'_s'$seed"

/home/lzhang726/anaconda3/envs/pytorch/bin/python -u run.py \
  --is_training 1 \
  --root_path /mnt/users/lzhang726/dataset/prediction/traffic/ \
  --data_path traffic.csv  \
  --model_id traffic_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 862 \
  --des 'Exp' \
  --H_order $H_order \
  --gpu 0 \
  --patience 20 \
  --train_epochs 50\
  --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/custom_abl/$m'_'$model_name'_'custom_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log
done
done
done

# export CUDA_VISIBLE_DEVICES=3
# model_name=FITS

# for H_order in 10
# do
# for seq_len in 720
# do
# # for m in 1
# # do
# # for seed in 514 1919 810 0 #114
# # do 
# for bs in 64 #256 #32 64 # 128 256
# do

# python -u run.py \
#   --is_training 1 \
#   --root_path /mnt/users/lzhang726/dataset/prediction/traffic/ \
#   --data_path traffic.csv  \
#   --model_id traffic_$seq_len'_'96 \
#   --model $model_name \
#   --data custom \
#   --features S\
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --H_order $H_order \
#   --gpu 0 \
#   --patience 20 \
#   --train_epochs 50\
#   --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/custom_abl/$m'_'$model_name'_'custom_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

# #   echo "Done $model_name'_'custom_$seq_len'_'96'_H'$H_order'_s'$seed"

# python -u run.py \
#   --is_training 1 \
#   --root_path /mnt/users/lzhang726/dataset/prediction/traffic/ \
#   --data_path traffic.csv  \
#   --model_id traffic_$seq_len'_'192 \
#   --model $model_name \
#   --data custom \
#   --features S\
#   --seq_len $seq_len \
#   --pred_len 192 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --H_order $H_order \
#   --gpu 0 \
#   --patience 20 \
#   --train_epochs 50\
#   --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/custom_abl/$m'_'$model_name'_'custom_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

# #   echo "Done $model_name'_'custom_$seq_len'_'192'_H'$H_order'_s'$seed"

# python -u run.py \
#   --is_training 1 \
#   --root_path /mnt/users/lzhang726/dataset/prediction/traffic/ \
#   --data_path traffic.csv  \
#   --model_id traffic_$seq_len'_'336 \
#   --model $model_name \
#   --data custom \
#   --features S\
#   --seq_len $seq_len \
#   --pred_len 336 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --H_order $H_order \
#   --gpu 0 \
#   --patience 20 \
#   --train_epochs 50\
#   --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/custom_abl/$m'_'$model_name'_'custom_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

# #   echo "Done $model_name'_'custom_$seq_len'_'336'_H'$H_order'_s'$seed"

# python -u run.py \
#   --is_training 1 \
#   --root_path /mnt/users/lzhang726/dataset/prediction/traffic/ \
#   --data_path traffic.csv  \
#   --model_id traffic_$seq_len'_'720 \
#   --model $model_name \
#   --data custom \
#   --features S\
#   --seq_len $seq_len \
#   --pred_len 720 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --H_order $H_order \
#   --gpu 0 \
#   --patience 20 \
#   --train_epochs 50\
#   --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/custom_abl/$m'_'$model_name'_'custom_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log
# done
# done
# done