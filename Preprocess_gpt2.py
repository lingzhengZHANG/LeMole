import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
class gpt2_Model(nn.Module):
    def __init__(self, configs):
        super(gpt2_Model, self).__init__()
        self.device = configs.gpu
        print(self.device)
        
        self.llama = GPT2Model.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        self.llama_tokenizer = GPT2Tokenizer.from_pretrained(configs.llm_ckp_dir)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.vocab_size = self.llama_tokenizer.vocab_size
        self.hidden_dim_of_llama = 768
        
        for name, param in self.llama.named_parameters():
            param.requires_grad = False

    def tokenizer(self, x):
        output = self.llama_tokenizer(x, return_tensors="pt")['input_ids'].to(self.device)
        result = self.llama.get_input_embeddings()(output)
        return result   
    
    def forecast(self, x_mark_enc):        
        # x_mark_enc: [bs x T x hidden_dim_of_llama]
        x_mark_enc = torch.cat([self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))], 0)
        # torch.Size([128, 26, 768])
        text_outputs = self.llama(inputs_embeds=x_mark_enc).last_hidden_state
        #torch.Size([128, 26, 768])
        text_outputs = text_outputs[:, -1, :]
        return text_outputs
    
    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)