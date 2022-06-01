from collections import defaultdict
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import sklearn.datasets
import torch
from torch import Tensor

class EpsTheta(torch.nn.Module):
    def __init__(self, T: int, num_hidden: int):
        super().__init__()
        self.T = T
        self.eps_theta = torch.nn.Sequential(
            torch.nn.Linear(in_features=IMG_DIM+1, out_features=hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=IMG_DIM))

    def forward(self, img: Tensor, t: int) -> Tensor:
        """img is a flat vector (possibly batched), t is a scalar"""
        t_scaled = self._scale_t(t)
        model_input = torch.concat([img, t_scaled], axis=-1)
        model_output = self.eps_theta(model_input)
        return model_output

    def _scale_t(self, t: int) -> Tensor:
        return torch.tensor([(t / self.T - 0.5) * 2])



# Dataset loading.
dataset = sklearn.datasets.load_digits()
data_np, labels_np = dataset['data'], dataset['target']
data_np = data_np[labels_np == 1]
del labels_np
data = torch.from_numpy(data_np.astype(np.float32))
data = ((data / 15) - 0.5) * 2  # Normalize between [-1, 1]
IMG_DIM = data.shape[1]

data_mean = torch.mean(data, axis=0)
imshow(data_mean)


def quality_assessment():
    x_i_list, _ = sample()
    x_gen = x_i_list[-1]
    return torch.linalg.norm(x_gen - data_mean)


# Hyperparameter ranges.

hparams = {
    'beta_1': list(np.exp(np.linspace(-10, -.8,num=15))),
    'beta_T_factor': [2, 10, 20, 100],
    'T': [2, 6, 10, 25, 100, 250, 1000],
    'hidden_size': [8, 16, 32, 64, 128],
    'num_layers': [1, 2, 3, 5, 10],
    'optim': ['adam', 'sgd'],
    'lr': [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
}

# Notebook content

def evaluate_hparam_choice(
    beta_1: float, beta_T: float, T: int, 
    hidden_size: int, num_layers: int, opt_name: str, lr: float) -> float:

    # Initialize network
    eps_theta = EpsTheta(T=T, hidden_size=hidden_size)

    # Alpha / beta vector computation.
    beta_t = torch.linspace(beta_1, beta_T, T)
    alpha_t = 1 - beta_t
    alpha_bar_t = torch.tensor([torch.prod(alpha_t[:s]) for s in range(1, T+1)])

    def beta(t: int) -> float:
        assert 0 < t <= T
        return beta_t[t-1]

    def alpha(t: int) -> float:
        assert 0 < t <= T
        return alpha_t[t-1]

    def alpha_bar(t: int) -> float:
        if t == 0: return alpha_bar_t[0]
        assert 0 < t <= T
        return alpha_bar_t[t-1]

    def train_loss(x_0: Tensor, t: int) -> Tensor:
        eps = torch.randn(x_0.shape)
        model_arg = alpha_bar(t).sqrt() * x_0 + (1 - alpha_bar(t)).sqrt() * eps
        eps_model_pred = eps_theta(img=model_arg, t=t)
        return torch.linalg.norm(eps - eps_model_pred) ** 2


    NUM_EPOCHS = 2000

    torch.manual_seed(124)
    eps_theta = EpsTheta(T=T)

    if opt_name == 'adam':
        opt = torch.optim.Adam(eps_theta.parameters(), lr=lr, betas=(0.9, 0.996))
    else:
        raise Error()

    for epoch in range(1, NUM_EPOCHS+1):
        opt.zero_grad()

        loss = torch.tensor(0.)
        for x_0 in data:
            t = torch.randint(1, T+1, size=(1,))
            loss += train_loss(x_0, t.item())

        loss /= len(data)
        loss.backward()
        opt.step()

        ll.append(loss.detach().item())

    def sample():
        normal = lambda: torch.randn(data_sample.shape)
        
        # line 1
        x_T = normal()
        x_i = x_T

        for i, t in tqdm(enumerate(range(T, 0, -1))):
            # line 3
            z = normal() if t > 1 else torch.zeros_like(data_sample)

            # line 4
            fac1 = 1 / alpha(t).sqrt()
            model_out = eps_theta(x_i, t)
            model_out_fac = (1 - alpha(t)) / (1 - alpha_bar(t)).sqrt()
            fac2 = x_i - model_out_fac * model_out
            sigma_t = beta(t).sqrt()
            x_i = fac1 * fac2 + sigma_t * z

        x_0 = x_i
        return x_0.detach() # line 6

    return quality_assessment().item()