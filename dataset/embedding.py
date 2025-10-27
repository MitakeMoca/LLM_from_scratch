import torch

vocab_size = 50257
output_dim = 256

# 输入词元 id，输出对应的嵌入向量
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
