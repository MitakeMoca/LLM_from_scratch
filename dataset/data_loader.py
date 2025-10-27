import tiktoken
from dataset.gpt_dataset import GPTDataset
from torch.utils.data import DataLoader


# data_loader 用于加载数据集
def create_dataloader(txt: str, batch_size=4, max_length=256, 
                      stride=128, shuffle=True, drop_last=True,
                      num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader