import torch

checkpoint = torch.load("/usrhomes/s1495/TIGER/Experiments/checkpoint/TIGER-small-on-EchoSet/last-v2.ckpt", map_location="cpu")
print(type(checkpoint))
print(checkpoint.keys())
