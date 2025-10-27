from data_loader import create_dataloader

# 也具有 encode 和 decode 方法，只不过是 BPE 编码
with open("../data/the-verdict.txt", "r", encoding="utf-8") as f:
    txt = f.read()

dataloader = create_dataloader(txt, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs: \n", inputs)
print("Targets: \n", targets)

inputs, targets = next(data_iter)
print("Inputs: \n", inputs)
print("Targets: \n", targets)

