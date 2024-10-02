import torch.nn as nn
import torch
from torch.optim import adam
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.activation = nn.ReLU()
        self.l2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        return x
    
train_data = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

epochs = 1000
m = MLP()
optim = adam.Adam(m.parameters(), lr=3e-4)
loss = None
for i in range(epochs):
    data, labels = next(iter(train_dataloader))
    pred = m(data)
    loss = nn.CrossEntropyLoss(reduction="mean")(pred, labels)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 100 == 0:
        # Accuracy calculation
        with torch.no_grad():  # Disable gradient calculation for evaluation
            predicted = torch.argmax(pred, dim=1)  # Get class with highest score
            correct = (predicted == labels).sum().item()  # Count correct predictions
            accuracy = correct / labels.size(0)  # Compute accuracy as a fraction of total samples
        print(f"loss: {loss}, accuracy: {accuracy}")

torch.save({
    'epoch': epochs,
    'model_state_dict': m.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'loss': loss
}, "./pretrained/mnist.pt")