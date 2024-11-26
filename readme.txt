LinearLLM		放model
data_factory	放data_provider
exp_long_term_forecasting和exp_basic	放exp
其余的放最外面

run.py 内部
root_path 改成你自己的数据位置
features S 单变量
enc_in，dec_in，c_out 保持1
datainfo_emb 数据集的描述gpt2特征，改成datainfo_emb_GPT2你放的位置