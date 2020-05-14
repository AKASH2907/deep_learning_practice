import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
seq_length = 28
inp_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
num_epochs = 5
batch_size = 64
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
	root='~/.pytorch-datasets/',
	train=True,
	transform = transforms.ToTensor(),
	download=True
)

test_dataset = torchvision.datasets.MNIST(
	root='~/.pytorch-datasets/',
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

# RNN
class RNN(nn.Module):
	def __init__(self, inp_size, hidden_size, num_layers, num_classes):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(inp_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		# Set initial hidden and cell states
		h0 = 
		c0 = 
		return out

model = RNN(inp_size, hidden_size, num_layers, num_classes).to(device)
	