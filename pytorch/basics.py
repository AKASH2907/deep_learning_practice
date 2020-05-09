import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Computational Graph
y = w * x + b

# Compute gradients
y.backward()

# print gradients
print(x.grad)
print(w.grad)
print(b.grad)
