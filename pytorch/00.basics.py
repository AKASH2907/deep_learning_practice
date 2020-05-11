import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# 1. Basic Autograd
x = torch.tensor(1.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# Computational Graph
y = w * x + b

# Compute gradients
y.backward()

# print gradients
print(x.grad)
print(w.grad)
print(b.grad)


# 2. Basic Autograd
x = torch.randn(10, 3)
y = torch.randn(10, 2)

linear = nn.Linear(3, 2)
print("w: ", linear.weight)
print("b: ", linear.bias)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)

loss = criterion(pred, y)
print("loss: ", loss.item())

# Backward pass
loss.backward()

# Grads
print("dL/dW: ", linear.weight.grad)
print("dL/dB: ", linear.bias.grad)

optimizer.step()

linear.weight.data.sub_(0.01 * linear.weight.grad.data)
linear.bias.data.sub_(0.01 * linear.bias.grad.data)

pred = linear(x)
loss = criterion(pred, y)
print("Loss after 1 optimization: ", loss.item())


# 3. Numpy to tensor
x = np.array([[1, 2], [3, 4]])

# numpy to torch
y = torch.from_numpy(x)

# torch to numpy
z = y.numpy()


# 4. Input pipeline for torchvision datasets
train_dataset = torchvision.datasets.CIFAR10(
    root="~/.pytorch-datasets/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

# Fetch data
image, label = train_dataset[0]
print(image.size())
print(label)

# Data Loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)

data_iteration = iter(train_loader)

# Mini-batch
images, label = data_iteration.next()

# Use of data loader
for images, label in train_loader:
	pass


# 5. Input pipeline for custom dataset

# Custom dataset
class CustomDataset(torch.utils.data.Dataset):
	def __init__(self):
		# File paths or list of names
		pass

	def __getitem__(self, index):
		# Read one data from file [e.g. PIL.Image.open, numpy.fromfile]
		# Preprocess data (torchvision.transforms)
		# Return data pair (image and label)
		pass

	def __len__(self):
		# Total size of dataset
		return 0

# Train Loader
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(
	dataset=custom_dataset,
	batch_size=64,
	shuffle=True
)


# 6. Pre-trained model loading
resnet = torchvision.models.resnet18(pretrained=True)

# Fine-tuning
for param in resnet.parameters():
	param.requires_grad = False

# Top layer for finetuning
resnet.fc = nn.Linear(resnet.fc.in_features, 50) # n_classes

# Forward pass
images = torch.randn(64, 3, 224, 224)
output = resnet(images)
print(output.size())


# 7. Save and load model
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save n load only model params (recommended))
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))