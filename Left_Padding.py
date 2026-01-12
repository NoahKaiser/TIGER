import torch
import torch.nn.functional as F

x = torch.tensor([1, 2, 3, 4])

# Pad 2 zeros on the LEFT
# For 1D: pad = (pad_left, pad_right)
y = F.pad(x, (2, 0))

print("x:", x)
print("y:", y)