import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyperparams
inp_size = 28*28
num_classes = 10
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
	data='~/.pytorch-datasets/',
	train=True,
	transform = transforms.ToTensor(),
	download=True
)

test_dataset = torchvision.datasets.MNIST(
	data='~/.pytorch-datasets/',
	train=False,
	transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
	dataset=train_dataset,
	batch_size=batch_size,
	shuffle=True
)

test_loader = torch.utils.data.DataLoader(
	dataset=test_dataset,
	batch_size=batch_size,
	shuffle=False
)

model = nn.Linear(inp_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Model training
total_step = len(train_loader)
for epoch in num_epochs:
	for (images, labels) in enumerate(train_loader):
		# Reshape images to batch size, input_size
		print(images.size())
		images = images.reshape(-1, inp_size)
		print(images.size())
		hghj
