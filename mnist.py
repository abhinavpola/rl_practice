# %%
import pandas as pd
from tinygrad import nn, Tensor
from tinygrad.nn.optim import SGD
import numpy as np

# %%
# Splitting training and test sets

train_df = pd.read_csv("datasets/mnist-in-csv/mnist_train.csv")
test_df = pd.read_csv("datasets/mnist-in-csv/mnist_test.csv")

x_train = train_df.iloc[:, 1:].values # pixels
y_train = train_df.iloc[:, 0].values # labels

x_test = test_df.iloc[:, 1:].values # pixels
y_test = test_df.iloc[:, 0].values # labels


# %%
def activation(x: Tensor) -> Tensor:
    return x.leakyrelu()

def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()

class Network:
    def __init__(self):
        self.layers = [nn.Linear(784, 128, bias=False), nn.Linear(128, 10, bias=False)]
        self.opt = SGD([self.layers[0].weight, self.layers[1].weight], lr=3e-4)

    def forward(self, x: Tensor):
        x = self.layers[0](x)
        x = activation(x)
        x = self.layers[1](x)

        return x

    def train(self, epochs: int):
        with Tensor.train():
            for i in range(epochs):
                # sample data
                sample = np.random.randint(0, x_train.shape[0], size=64)
                batch = Tensor(x_train[sample], requires_grad=False)
                labels = Tensor(y_train[sample])
                pred = self.forward(batch)
                loss = sparse_categorical_crossentropy(pred, labels)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                Y = pred.argmax(axis=-1)
                acc = (Y == labels).mean()
                if i%100 == 0:
                    print(f"Step {i+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")


net = Network()
#print(x_train[].shape)
net.train(1000)
