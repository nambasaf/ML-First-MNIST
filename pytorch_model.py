import torch
from torch import nn
from torchvision import datasets, transforms
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST(root=".", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=".", train=False, download=True, transform=transform)

# Create dataloaders for batching and shuffling data(shuffling the train data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):    # CNN
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3),  # 1 input channel (grayscale), 32 filters, 3x3 kernel
            nn.ReLU(),
            nn.MaxPool2d(2),      # halves image size: 28x28 -> 14x14

            nn.Conv2d(32, 64, 3), # 64 filters, again 3x3
            nn.ReLU(),
            nn.MaxPool2d(2),      # 14x14 -> 7x7

            nn.Flatten(),         # flatten tensor from [64, 7, 7] -> [64*7*7]
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
print("starting the training loop\n")
for epoch in range(3):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluate Model
correct = 0
total = 0
print("starting the testing loop\n")

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()          

print("Pytorch model accuracy:", correct / total)



        

