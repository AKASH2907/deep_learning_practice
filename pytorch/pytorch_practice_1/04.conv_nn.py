import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
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

class ConvNet(nn.Module):
	def __init__(self, num_classes=10):
		super(ConvNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.fc = nn.Linear(7*7*32, num_classes)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)

		return out


model = ConvNet(num_classes).to(device)

# Loss n optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Model training
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		# Reshape images to batch size, input_size
		images = images.to(device)
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
torch.save(model.state_dict(), 'conv.ckpt')
