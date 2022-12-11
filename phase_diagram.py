import torch
from torch import nn
import torch.nn.functional as F
import torch_optimizer as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class ParallelReLUModels(nn.Module):
    def __init__(self, averaging_dims, density_dims, importance_dims, feature_size, activation_size):
        super().__init__()

        self.W = nn.Parameter(torch.empty(averaging_dims, density_dims, importance_dims, activation_size, feature_size))
        nn.init.xavier_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(averaging_dims, density_dims, importance_dims, feature_size))
    
    def forward(self, x):
        return F.relu((x @ torch.transpose(self.W, -1, -2) @ self.W) + self.b.unsqueeze(3))

def generate_sparse_data(averaging_dims, density_dims, importance_dims, feature_size, batch_size, densities):
    mask = torch.rand(averaging_dims, density_dims, importance_dims, batch_size, feature_size) < densities[:, None, None, None]
    return torch.rand(averaging_dims, density_dims, importance_dims, batch_size, feature_size) * mask

def importance_weighted_MSE(x_prime, x, importances):
    return torch.mean(torch.square(x_prime - x) * importances.unsqueeze(1), dim=-1)

def train(model, averaging_dims, densities, rel_importances, feature_size, batch_size):
    loss_fn = importance_weighted_MSE
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=0.001)
    density_dims = len(densities)
    importance_dims = len(rel_importances)

    # create importances from relative importances
    importances = torch.stack((torch.ones(importance_dims), rel_importances), dim=1)

    iterations = 5000
    for t in range(iterations):
        data = generate_sparse_data(averaging_dims, density_dims, importance_dims, feature_size, batch_size, densities)

        pred = model(data)
        losses = loss_fn(pred, data, importances)
        loss = torch.mean(losses)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (t % 100) == 0:
            print(f"Iteration {t+1}\n-------------------------------")
            print(f"loss: {loss.item():>7f}")

    return losses

# x coordinate makes color go from red to blue, y makes color get whiter
def color_map_2d(x, y):
    # clamp values to 0-1
    x, y = max(min(x, 1), 0), max(min(y, 1), 0)

    # red and blue
    min_hue, max_hue = 2/3, 1
    hue = ((max_hue - min_hue) * x) + min_hue

    return mpl.colors.hsv_to_rgb((hue, y, 1))

def create_plots(model, losses, densities, rel_importances, feature_size):
    W = model.W.detach()
    weighting = F.normalize(1/losses, p=1, dim=0)

    feature_norms = torch.linalg.norm(W, dim=-2)

    mask = (1 - torch.eye(feature_size))[None, None, :, :]
    W_normalized = W / torch.linalg.norm(W, dim=-2, keepdim=True)
    feature_superpositions = torch.sum(torch.square((W_normalized.transpose(-1, -2) @ W) * mask), dim=-1)

    feature_norms_avg = torch.sum(weighting * feature_norms, dim=0)
    feature_superpositions_avg = torch.mean(weighting * feature_superpositions, dim=0)

    image = np.zeros((len(densities), len(rel_importances), 3))
    for d in range(len(densities)):
        for r in range(len(rel_importances)):
            image[d, r, :] = color_map_2d(feature_superpositions_avg[d, r, 0], feature_norms_avg[d, r, 0])

    plt.imshow(image)
    plt.matshow(feature_norms_avg[:, :, 0])
    plt.matshow(feature_superpositions_avg[:, :, 0])
    plt.matshow(feature_norms_avg[:, :, 1])
    plt.matshow(feature_superpositions_avg[:, :, 1])
    plt.show()


# reproducible results
torch.manual_seed(1)
np.random.seed(1)

feature_size = 2
activation_size = 1
batch_size = 1024
averaging_dims = 10
# density goes from 1/100 to 1
densities = 10 ** torch.linspace(-2, 0, 20)
# relative importance goes from 1/10X to 10X
rel_importances = 10 ** torch.linspace(-1, 1, 20)


model = ParallelReLUModels(averaging_dims, len(densities), len(rel_importances), feature_size, activation_size)

losses = train(model, averaging_dims, densities, rel_importances, feature_size, batch_size)
print("Done!")

create_plots(model, losses, densities, rel_importances, feature_size)

#m = np.zeros((100, 100, 3))
#for i in range(100):
#    for j in range(100):
#        m[i, j, :] = color_map_2d(i/99, j/99)

#plt.imshow(m)
#plt.show()

