import numpy as np
import torch

a = torch.Tensor([0, 2, 80, 80, 80, 68, 0, 0, 0]).numpy()
b = torch.Tensor([1, 2, 80, 80, 80, 68, 0, 0, 0]).numpy()
m = torch.Tensor([0, 1, 1, 1, 1, 1, 0, 0, 0]).numpy()

print(a[m == 1])
print(b[m == 1])
r = (a[m == 1] == b[m == 1])
print(r)
print(np.sum(r==False))
print(1/3)


