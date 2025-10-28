import tiktoken
import torch
from models.GPT_model import GPTModel
from util.generate import generate_text_simple

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")
txt = "Hello, I am"

encoded = tokenizer.encode(txt)
print(encoded)

# 添加 batch 维度
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape: ", encoded_tensor.shape)

# GPT2 模型配置
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

model = GPTModel(GPT_CONFIG_124M)

# 评估模式，将不会使用类似于 dropout 这种模块
model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output: ", out)
print("Output length: ", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print("decoded text: ", decoded_text)