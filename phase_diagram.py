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

def generate_sparse_data(averaging_dims, importance_dims, feature_size, batch_size, densities, device):
    mask = torch.rand(averaging_dims, len(densities), importance_dims, batch_size, feature_size, device=device) < densities[:, None, None, None]
    return torch.rand(averaging_dims, len(densities), importance_dims, batch_size, feature_size, device=device) * mask

def importance_weighted_MSE(x_prime, x, importances):
    return torch.mean(torch.square(x_prime - x) * importances.unsqueeze(1), dim=[-1, -2])

def train(model, averaging_dims, densities, rel_importances, feature_size, batch_size, device):
    loss_fn = importance_weighted_MSE
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=0.001)
    density_dims = len(densities)
    importance_dims = len(rel_importances)

    # create importances from relative importances
    importances = torch.stack((torch.ones(importance_dims, device=device), rel_importances), dim=1)

    iterations = 5000
    for t in range(iterations):
        data = generate_sparse_data(averaging_dims, importance_dims, feature_size, batch_size, densities, device)

        pred = model(data)
        loss = torch.mean(loss_fn(pred, data, importances))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (t % 100) == 0:
            print(f"loss: {loss.item():>7f}")
            print(f"Iteration {t+1}\n-------------------------------")

# x coordinate makes color go from red to blue, y makes color get whiter
def color_map_2d(x, y):
    # clamp values to 0-1
    x, y = max(min(x, 1), 0), max(min(y, 1), 0)

    # red and blue
    min_hue, max_hue = 2/3, 1
    hue = ((max_hue - min_hue) * x) + min_hue

    return mpl.colors.hsv_to_rgb((hue, y, 1))

def create_plots(model, densities, rel_importances, feature_size, averaging_dims):
    model = model.to('cpu')
    W = model.W.detach()

    test_batch = generate_sparse_data(averaging_dims, len(rel_importances), feature_size, 8192, densities.cpu(), 'cpu')
    importances = torch.stack((torch.ones(len(rel_importances)), rel_importances.cpu()), dim=1)

    model.eval()
    with torch.no_grad():
        test_preds = model(test_batch)

    # use losses to weight result average (lower loss gets weighted higher)
    losses = importance_weighted_MSE(test_batch, test_preds, importances)
    weighting = F.normalize(1 / (losses.cpu() ** 2), p=1, dim=0).unsqueeze(-1)

    feature_norms = torch.linalg.norm(W, dim=-2)

    mask = (1 - torch.eye(feature_size))[None, None, :, :]
    W_normalized = W / torch.linalg.norm(W, dim=-2, keepdim=True)
    feature_superpositions = torch.sum(torch.square((W_normalized.transpose(-1, -2) @ W) * mask), dim=-1)

    feature_norms_avg = torch.sum(feature_norms * weighting, dim=0)
    feature_superpositions_avg = torch.sum(feature_superpositions * weighting, dim=0)

    image = np.zeros((len(densities), len(rel_importances), 3))
    for d in range(len(densities)):
        for r in range(len(rel_importances)):
            image[d, r, :] = color_map_2d(feature_superpositions_avg[d, r, 0], feature_norms_avg[d, r, 0])

    plt.imshow(image, origin='lower')
    plt.show()


# reproducible results
torch.manual_seed(1)
np.random.seed(1)

device = 'cuda'
feature_size = 2
activation_size = 1
batch_size = 1024
averaging_dims = 10
# density goes from 1/100 to 1
densities = 10 ** torch.linspace(-2, 0, 50, device=device)
# relative importance goes from 1/10X to 10X
rel_importances = 10 ** torch.linspace(-1, 1, 50, device=device)


model = ParallelReLUModels(averaging_dims, len(densities), len(rel_importances), feature_size, activation_size).to(device)

train(model, averaging_dims, densities, rel_importances, feature_size, batch_size, device)
print("Done!")

create_plots(model, densities, rel_importances, feature_size, averaging_dims)

