from dataset.data_loader import create_dataloader
from dataset.embedding import embedding_layer
from dataset.pos_embedding import pos_embedding_layer
from attention.multi_head_attention import MultiHeadAttention
import torch

torch.manual_seed(123)

# 也具有 encode 和 decode 方法，只不过是 BPE 编码
with open("./data/the-verdict.txt", "r", encoding="utf-8") as f:
    txt = f.read()

# 加载数据，这一步包含了分词，token -> token ID 任务
dataloader = create_dataloader(txt, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs: \n", inputs)
print("\nInput Shape: \n", inputs.shape)
print("Targets: \n", targets)

# token ID -> token embeddings 
token_embeddings = embedding_layer(inputs)
print(token_embeddings)
print(token_embeddings.shape)

# pos embeddings
pos_embeddings = pos_embedding_layer(torch.arange(4))
print(pos_embeddings.shape)

# 位置嵌入融合词嵌入
input_embeddings = token_embeddings + pos_embeddings
print("Input_embeddings: \n", input_embeddings)
print("Input Embeddings Shape: \n", input_embeddings.shape)

# 模拟 batch 输入示例
batch = input_embeddings
batch_size, context_length, d_in = batch.shape
d_out = 2

mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape: ", context_vecs.shape)