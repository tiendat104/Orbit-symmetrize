# pylint: disable=line-too-long,missing-function-docstring,missing-class-docstring,missing-module-docstring,invalid-name,no-member
import os
import random
import pdb
import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from emlp_pytorch.datasets import ParticleInteraction
from emlp_pytorch.groups import Lorentz


SEED = 42


def compute_invariant_label(x):
    # this function is for debugging purpose, can be removed
    # x: [b, n, d]
    p1, p2, p3, p4 = x.permute(1, 0, 2)  # [n, b, d]
    eng = torch.diag(torch.tensor([1., -1., -1., -1.], device=x.device))
    def dot(v1, v2):
        return ((v1 @ eng) * v2).sum(-1)
    Le = p1[:, :, None] * p3[:, None, :] - (dot(p1, p3) - dot(p1, p1))[:, None, None] * eng
    Lmu = (p2 @ eng)[:, :, None] * (p4 @ eng)[:, None, :] - (dot(p2, p4) - dot(p2, p2))[:, None, None] * eng
    label = 4 * (Le * Lmu).sum(-1).sum(-1)
    return label[..., None]


def orbit_separating_invariant(x):
    # x: [b, n, d]
    if x.ndim == 2:
        return orbit_separating_invariant(x[None, :, :])[0]
    _, _, d = x.shape
    assert d == 4
    # lambda [d, d]
    l = torch.tensor([
        [-1, 0, 0, 0],
        [0, +1, 0, 0],
        [0, 0, +1, 0],
        [0, 0, 0, +1]
    ], dtype=x.dtype, device=x.device)
    # compute lambda-gram matrix [b, n, n]
    return torch.einsum('bij,jk,blk->bil', x, l, x)


def sample_equivariant_noise(x, m=16):
    # x: [b, n, d] -> z: [b, m, d]
    b, n, d = x.shape
    assert (n, d) == (4, 4)
    # invariant [b, m, n] = z @ lambda @ x.T
    invariant = torch.rand(b, m, n, dtype=x.dtype, device=x.device)
    # lambda [d, d]
    l = torch.tensor([
        [-1, 0, 0, 0],
        [0, +1, 0, 0],
        [0, 0, +1, 0],
        [0, 0, 0, +1]
    ], dtype=x.dtype, device=x.device)
    # z = invariant @ (lambda @ x.T)^-1
    lx = torch.einsum('id,bnd->bin', l, x)
    z = torch.einsum('bmn,bnd->bmd', invariant, torch.linalg.inv(lx))
    # compute lambda-gram matrix [b, m, m]
    z_invariant = torch.einsum('bij,jk,blk->bil', z, l, z)
    # compute invariant scale [b, m, 1]
    z_scale = z_invariant.abs().sum(-1, keepdim=True).expand(b, m, d)
    # normalize
    z = z / z_scale.clamp(min=1e-6)
    return z


class ScalarMLP(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.group = Lorentz().to(device)
        x_dim = 4
        z_dim = 128
        out_dim = 4
        dim = 500
        dropout = 0.1
        self.net = nn.Sequential(
            nn.Linear((x_dim + z_dim) * (x_dim + z_dim), dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, (x_dim + z_dim) * out_dim)
        ).to(device)
        self.z_dim = z_dim
        self.out_dim = out_dim

    def _invariant(self, x):
        # [b, n, d] -> [b, n, m]
        b, n, _ = x.shape
        x = orbit_separating_invariant(x).reshape(b, n * n)
        return self.net(x).reshape(b, n, self.out_dim)

    def _forward(self, x, z):
        # [b, n, d], [b, m, d] -> [b, n + m, n]
        xz = torch.cat([x, z], dim=1)  # [b, n + m, d]
        w = self._invariant(xz)  # [b, n + m, n]
        output = torch.einsum('bnd,bno->bod', xz, w)  # [b, n, d]
        return output

    def forward(self, x, test_equivariance=False):
        # [b, n * d] -> [b, m * d]
        bsize = x.shape[0]
        assert x.shape == (bsize, 16)
        x = x.reshape(bsize, 4, 4)  # [b, n, d]
        z = sample_equivariant_noise(x, self.z_dim)  # [b, m, d]
        output = self._forward(x, z)  # [b, n, d]
        output = output.reshape(bsize, 16)  # [b, n * d]
        if test_equivariance:
            assert not self.training
            with torch.no_grad():
                lorentz = self.group.samples(bsize)
                lorentz_inv = torch.linalg.inv(lorentz)
                output1 = self._forward(x, z)
                transformed_x = torch.einsum('bnj,bij->bni', x, lorentz)
                transformed_z = torch.einsum('bnj,bij->bni', z, lorentz)
                output2 = self._forward(transformed_x, transformed_z)
                output3 = torch.einsum('bnj,bij->bni', output2, lorentz_inv)
                equivariance_error = torch.nn.functional.mse_loss(output1, output3) / torch.nn.functional.mse_loss(output1, output2)
                return output, equivariance_error
        return output


def main():
    # reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    NUM_TRAIN = 10000
    NUM_VAL = 1000
    NUM_TEST = 5000
    BSIZE = 500
    LR = 3e-3
    device = torch.device('cuda:0')
    num_epochs = min(int(900000 / NUM_TRAIN), 1000)
    print(f"num_epochs: {num_epochs}")

    train_dataset = ParticleInteraction(NUM_TRAIN)
    val_dataset = ParticleInteraction(NUM_VAL)
    test_dataset = ParticleInteraction(NUM_TEST)

    train_loader = DataLoader(train_dataset, batch_size=BSIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BSIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BSIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = ScalarMLP(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm.tqdm(train_loader)
        for x, y in pbar:
            x = x.to(device)  # [b, n * d]
            y = y.to(device)
            bsize = x.shape[0]
            assert x.shape == (bsize, 16)

            target = orbit_separating_invariant(torch.eye(4, dtype=x.dtype, device=device))
            target = target[None, :, :].expand(bsize, 4, 4)
            output = model(x)  # [b, n * d]
            output = output.reshape(bsize, 4, 4)  # [b, n, d]
            output = orbit_separating_invariant(output)

            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"train: Epoch [{epoch}/{num_epochs}]")
            pbar.set_postfix(loss=loss.item())

        model.eval()
        pbar = tqdm.tqdm(val_loader)
        with torch.no_grad():
            for x, y in pbar:
                x = x.to(device)  # [b, n * d]
                y = y.to(device)
                bsize = x.shape[0]
                assert x.shape == (bsize, 16)

                target = orbit_separating_invariant(torch.eye(4, dtype=x.dtype, device=device))
                target = target[None, :, :].expand(bsize, 4, 4)
                output, equivariance_error = model(x, True)  # [b, n * d]
                output = output.reshape(bsize, 4, 4)  # [b, n, d]
                output = orbit_separating_invariant(output)

                l1_loss = torch.nn.functional.l1_loss(output, target)
                mse_loss = torch.nn.functional.mse_loss(output, target)

                pbar.set_description(f"val: Epoch [{epoch}/{num_epochs}]")
                pbar.set_postfix(l1_loss=l1_loss.item(), mse_loss=mse_loss.item(), equivariance_error=equivariance_error.item())

    pdb.set_trace()


if __name__ == '__main__':
    main()
