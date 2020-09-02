import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyperparams
inp_size = 28*28
num_classes = 10
num_epochs = 2
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

model = nn.Linear(inp_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Model training
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		# Reshape images to batch size, input_size		
		images = images.reshape(-1, inp_size)
		
		# Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1)%100==0:
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
		   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Model test
# No gradients computation

with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.reshape(-1, inp_size)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()

	print("Accuracy of the model on the 10000 test images: {} %".format(100 * correct // total))

# Save the model checkpoint
torch.save(model.state_dict(), 'log_reg.ckpt')
