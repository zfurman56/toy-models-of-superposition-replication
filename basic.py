import torch
from torch import nn
import torch.nn.functional as F
import torch_optimizer as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class LinearModel(nn.Module):
    def __init__(self, feature_size, activation_size):
        super().__init__()

        self.W = nn.Parameter(torch.empty(activation_size, feature_size))
        nn.init.xavier_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(feature_size))
    
    def forward(self, x):
        return (x @ self.W.T @ self.W) + self.b

class ReLUModel(nn.Module):
    def __init__(self, feature_size, activation_size):
        super().__init__()

        self.W = nn.Parameter(torch.empty(activation_size, feature_size))
        nn.init.xavier_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(feature_size))
    
    def forward(self, x):
        return F.relu((x @ self.W.T @ self.W) + self.b)

def generate_sparse_data(feature_size, batch_size, density):
    mask = torch.rand(batch_size, feature_size) < density
    return torch.rand(batch_size, feature_size) * mask

def importance_weighted_MSE(x_prime, x):
    batch_size = x.size(0)
    feature_size = x.size(1)

    # importance weight by 0.9^i where i is the feature number
    importance = 0.9 ** torch.arange(0, feature_size)
    return torch.mean(importance * torch.square(x_prime - x))

def train(model, feature_size, batch_size, density):
    loss_fn = importance_weighted_MSE
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=0.001)

    iterations = 10000
    for t in range(iterations):
        data = generate_sparse_data(feature_size, batch_size, density)

        pred = model(data)
        loss = loss_fn(pred, data)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (t % 100) == 0:
            print(f"Iteration {t+1}\n-------------------------------")
            print(f"loss: {loss.item():>7f}")

def create_plots(model, feature_size):
    W = model.W.detach()
    b = model.b.detach()

    feature_norms = torch.linalg.norm(W, dim=0)

    mask = (1 - torch.eye(feature_size))
    W_normalized = W / torch.linalg.norm(W, dim=0, keepdim=True)
    feature_superpositions = torch.sum(torch.square((W_normalized.T @ W) * mask), dim=1)
    feature_colors = [mpl.colormaps['inferno'](x.item()) for x in feature_superpositions]

    plt.matshow(W.T @ W, cmap='bwr', vmin=-1, vmax=1)
    plt.matshow(b.view([b.size(0), 1]), cmap='bwr', vmin=-1, vmax=1)
    plt.figure()
    plt.barh(np.arange(0, feature_size), feature_norms, color=feature_colors)
    plt.gca().invert_yaxis()
    plt.show()


# reproducible results
torch.manual_seed(1)
np.random.seed(1)

feature_size = 80
activation_size = 20
density = 0.1
batch_size = 1024

model = ReLUModel(feature_size, activation_size)

train(model, feature_size, batch_size, density)
print("Done!")

create_plots(model, feature_size)

