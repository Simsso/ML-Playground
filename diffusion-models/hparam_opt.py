import itertools
import numpy as np
import random
import sklearn.datasets
from tqdm import tqdm
import torch
from torch import Tensor


class EpsTheta(torch.nn.Module):
    def __init__(self, T: int, hidden_size: int):
        super().__init__()
        self.T = T
        self.eps_theta = torch.nn.Sequential(
            torch.nn.Linear(in_features=IMG_DIM + 1, out_features=hidden_size),
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
data_sample = data[0]
IMG_DIM = data.shape[1]

data_mean = torch.mean(data, axis=0)

# Hyperparameter ranges.

hparam_ranges = {
    'beta_1': list(np.exp(np.linspace(-10, -.8, num=15))),
    'beta_T_factor': [2, 10, 20, 100],
    'T': [2, 6, 10, 25, 100, 250, 1000],
    'hidden_size': [8, 16, 32, 64, 128],
    'opt_name': ['adam'],  # TODO: Add SGD.
    'lr': [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
    'num_epochs': [100],
}


# Notebook content

def evaluate_hparam_choice(
        beta_1: float, beta_T_factor: float, T: int,
        hidden_size: int, opt_name: str, lr: float, num_epochs: int) -> float:
    # Alpha / beta vector computation.
    beta_T = beta_1 * beta_T_factor
    beta_t = torch.linspace(beta_1, beta_T, T)
    alpha_t = 1 - beta_t
    alpha_bar_t = torch.tensor([torch.prod(alpha_t[:s]) for s in range(1, T + 1)])

    def beta(t: int) -> float:
        assert 0 < t <= T
        return beta_t[t - 1]

    def alpha(t: int) -> float:
        assert 0 < t <= T
        return alpha_t[t - 1]

    def alpha_bar(t: int) -> float:
        if t == 0: return alpha_bar_t[0]
        assert 0 < t <= T
        return alpha_bar_t[t - 1]

    def train_loss(x_0: Tensor, t: int) -> Tensor:
        eps = torch.randn(x_0.shape)
        model_arg = alpha_bar(t).sqrt() * x_0 + (1 - alpha_bar(t)).sqrt() * eps
        eps_model_pred = eps_theta(img=model_arg, t=t)
        return torch.linalg.norm(eps - eps_model_pred) ** 2

    torch.manual_seed(124)
    eps_theta = EpsTheta(T=T, hidden_size=hidden_size)

    if opt_name == 'adam':
        opt = torch.optim.Adam(eps_theta.parameters(), lr=lr, betas=(0.9, 0.996))
    else:
        raise Exception()

    for epoch in range(1, num_epochs + 1):
        opt.zero_grad()

        loss = torch.tensor(0.)
        for x_0 in data:
            t = torch.randint(1, T + 1, size=(1,))
            loss += train_loss(x_0, t.item())

        loss /= len(data)
        loss.backward()
        opt.step()

    def sample():
        normal = lambda: torch.randn(data_sample.shape)

        # line 1
        x_T = normal()
        x_i = x_T

        for i, t in enumerate(range(T, 0, -1)):
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
        return x_0.detach()  # line 6

    def quality_assessment():
        # TODO: average across multiple samples
        x_gen = sample()
        return torch.linalg.norm(x_gen - data_mean)

    return quality_assessment().item()


def main():
    hparam_names = sorted(hparam_ranges.keys())
    print(hparam_names)
    hparam_values = [hparam_ranges[n] for n in hparam_names]
    hparam_combinations = [dict(zip(hparam_names, v)) for v in itertools.product(*hparam_values)]
    random.shuffle(hparam_combinations)

    with open('hparam_search_results.csv', 'a') as f:
        f.write(','.join(hparam_names + ['data_mse']) + '\n')
        for hparams in tqdm(hparam_combinations):
            try:
                quality = evaluate_hparam_choice(**hparams)
            except Exception as e:
                print(hparams)
                print(e)
                quality = 'n/a'
            out_values = map(str, [hparams[n] for n in hparam_names] + [quality])
            f.write(','.join(out_values) + '\n')
            f.flush()


if __name__ == '__main__':
    main()
