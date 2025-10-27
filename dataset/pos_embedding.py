import torch

context_length = 4
output_dim = 256

pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)