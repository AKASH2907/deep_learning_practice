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
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

		# Forward propagate to LSTM
		out, _ = self.lstm(x, (h0, c0)) # out: tenosr shape (batch_size, seq_length, hidden_size)

		# Decode hidden state of last time step
		out = self.fc(out[:, -1, :])

		return out

model = RNN(inp_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.reshape(-1, seq_length, inp_size).to(device)
		labels = labels.to(device)

		# Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)

		# backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1) % 100 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
				   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# Test the model
model.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.reshape(-1, seq_length, inp_size).to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted==labels).sum().item()

	print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# save the model checkpoint
torch.save(model.state_dict(), 'rnn.ckpt')


