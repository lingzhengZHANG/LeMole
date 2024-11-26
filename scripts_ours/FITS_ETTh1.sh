export CUDA_VISIBLE_DEVICES=0
model_name=FITS

for H_order in 6
do
for seq_len in 360
do
# for m in 1
# do
# for seed in 514 1919 810 0 #114
# do 
for bs in 64 #256 #32 64 # 128 256
do

python -u run.py \
  --is_training 1 \
  --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --H_order $H_order \
  --gpu 0 \
  --patience 20 \
  --train_epochs 50\
  --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

#   echo "Done $model_name'_'Etth1_$seq_len'_'96'_H'$H_order'_s'$seed"

python -u run.py \
  --is_training 1 \
  --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --H_order $H_order \
  --gpu 0 \
  --patience 20 \
  --train_epochs 50\
  --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

#   echo "Done $model_name'_'Etth1_$seq_len'_'192'_H'$H_order'_s'$seed"

python -u run.py \
  --is_training 1 \
  --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --H_order $H_order \
  --gpu 0 \
  --patience 20 \
  --train_epochs 50\
  --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

#   echo "Done $model_name'_'Etth1_$seq_len'_'336'_H'$H_order'_s'$seed"

python -u run.py \
  --is_training 1 \
  --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --H_order $H_order \
  --gpu 0 \
  --patience 20 \
  --train_epochs 50\
  --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log
done
done
done

export CUDA_VISIBLE_DEVICES=0
model_name=FITS

for H_order in 6
do
for seq_len in 360
do
# for m in 1
# do
# for seed in 514 1919 810 0 #114
# do 
for bs in 64 #256 #32 64 # 128 256
do

python -u run.py \
  --is_training 1 \
  --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features S\
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --H_order $H_order \
  --gpu 0 \
  --patience 20 \
  --train_epochs 50\
  --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

#   echo "Done $model_name'_'Etth1_$seq_len'_'96'_H'$H_order'_s'$seed"

python -u run.py \
  --is_training 1 \
  --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'192 \
  --model $model_name \
  --data ETTh1 \
  --features S\
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 1 \
  --des 'Exp' \
  --H_order $H_order \
  --gpu 0 \
  --patience 20 \
  --train_epochs 50\
  --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

#   echo "Done $model_name'_'Etth1_$seq_len'_'192'_H'$H_order'_s'$seed"

python -u run.py \
  --is_training 1 \
  --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'336 \
  --model $model_name \
  --data ETTh1 \
  --features S\
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 1 \
  --des 'Exp' \
  --H_order $H_order \
  --gpu 0 \
  --patience 20 \
  --train_epochs 50\
  --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

#   echo "Done $model_name'_'Etth1_$seq_len'_'336'_H'$H_order'_s'$seed"

python -u run.py \
  --is_training 1 \
  --root_path /mnt/users/lzhang726/dataset/prediction/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'720 \
  --model $model_name \
  --data ETTh1 \
  --features S\
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --H_order $H_order \
  --gpu 0 \
  --patience 20 \
  --train_epochs 50\
  --itr 1 --batch_size $bs --learning_rate 0.0005 #| tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log
done
done
done