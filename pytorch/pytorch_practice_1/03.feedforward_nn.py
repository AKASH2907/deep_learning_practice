import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
inp_size = 784
hidden_units = 500
num_classes = 10
num_epochs = 5
batch_size = 64
learning_rate = 0.001

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


# Fully connected NN w/ oen hidden layer
class NeuralNet(nn.Module):
	def __init__(self, inp_size, hidden_units, num_classes):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(inp_size, hidden_units)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_units, num_classes)

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)

		return out


model = NeuralNet(inp_size, hidden_units, num_classes).to(device)

# Loss n optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Model training
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		# Reshape images to batch size, input_size
		images = images.reshape(-1, 28*28).to(device)
		labels = labels.to(device)

		# Forward pass
		output = model(images)
		loss = criterion(output, labels)

		# Backward n optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1) % 100 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
				   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# Test model
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.reshape(-1, 28*28).to(device)
		labels = labels.to(device)
		output = model(images)
		_, predict = torch.max(output.data, 1)
		total += labels.size(0)
		correct += (predict == labels).sum()

	print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'feedforward.ckpt')
