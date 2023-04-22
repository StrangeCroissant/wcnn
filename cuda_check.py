import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"

print(torch.cuda.is_available())
# print(device)
