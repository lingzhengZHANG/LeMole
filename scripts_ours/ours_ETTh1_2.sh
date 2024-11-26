export CUDA_VISIBLE_DEVICES=1
model_name=LinearLLM
seq_len=512
for t_dim in 1 2 3 4 5;do
for seq_len in 512 ;do
  for pred_len in  720  ;do
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