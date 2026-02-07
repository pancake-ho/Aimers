import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: [{DEVICE}]")