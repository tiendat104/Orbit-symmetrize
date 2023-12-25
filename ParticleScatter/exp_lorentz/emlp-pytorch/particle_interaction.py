# pylint: disable=line-too-long,missing-function-docstring,missing-class-docstring,missing-module-docstring,invalid-name,no-member,not-callable
import os
import argparse
import random
import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed

from emlp_pytorch.datasets import ParticleInteraction
from emlp_pytorch.groups import Lorentz


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # experiment arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device_id', type=int, default=0)

    # data arguments
    parser.add_argument('--num_train', type=int, default=10000)
    parser.add_argument('--num_val', type=int, default=1000)
    parser.add_argument('--num_test', type=int, default=5000)

    # model arguments
    parser.add_argument('--model', type=str, default='lps', choices=['mlp', 'mlp_aug', 'lps', 'scalar_mlp', 'canonical'])

    # probabilistic symmetrization arguments
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--eval_sample_size', type=int, default=1000)

    # training arguments
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--gradient_clip', type=float, default=1.0)

    return parser


def get_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser('Lorentz Symmetrization')
    parser = add_args(parser)
    args = parser.parse_args()
    return args


def configure_device(args):
    return torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')


def configure_data(args):
    device = configure_device(args)
    # setup dataset
    with FixedNumpySeed(2021), FixedPytorchSeed(2021):
        train_dataset = ParticleInteraction(args.num_train)
        val_dataset = ParticleInteraction(args.num_val)
        test_dataset = ParticleInteraction(args.num_test)
        # random Lorentz transformation to detect non-invariant cheating solution
        val_lorentz = Lorentz().to(device).samples(args.batch_size)  # [b, d, d]
        test_lorentz = Lorentz().to(device).samples(args.batch_size) @ Lorentz().to(device).samples(args.batch_size) @ Lorentz().to(device).samples(args.batch_size) @ Lorentz().to(device).samples(args.batch_size)  # [b, d, d]
    # setup loaders
    train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, args.num_train), shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=min(args.batch_size, args.num_val), shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, args.num_test), shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader, val_lorentz, test_lorentz, train_dataset.stats


def configure_model(args, train_stats):
    device = configure_device(args)
    # setup model
    if not args.model == 'lps':
        assert args.sample_size == 1 and args.eval_sample_size == 1
    if args.model == 'mlp':
        model = SymmetrizedMLP(device, use_symmetrization=False, use_aug=False)
    elif args.model == 'mlp_aug':
        model = SymmetrizedMLP(device, use_symmetrization=False, use_aug=True)
    elif args.model == 'lps':
        model = SymmetrizedMLP(device, use_symmetrization=True, use_aug=False)
    elif args.model == 'canonical':
        model = CanonicalizedMLP(device)
    elif args.model == 'scalar_mlp':
        model = ScalarMLP(device)
    else:
        raise NotImplementedError(f'Unknown model: {args.model}')
    model = Standardize(model, train_stats)
    print(f"Model: {args.model}, number of parameters: {sum(p.numel() for p in model.parameters())}")
    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer


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


def lorentz_inverse(x):
    # x: [b, d, channels=d] -> x_inv: [b, d, d]
    # lambda [d, d]
    l = torch.tensor([
        [-1, 0, 0, 0],
        [0, +1, 0, 0],
        [0, 0, +1, 0],
        [0, 0, 0, +1]
    ], dtype=x.dtype, device=x.device)
    # compute inverse
    x_inv = torch.einsum('ij,bkj,kl->bil', l, x, l)
    return x_inv


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


def sample_equivariant_noise(x, invariant):
    # x: [b, n, d] -> z: [b, m, d]
    b, n, d = x.shape
    m = invariant.shape[1]
    assert (n, d) == (4, 4)
    assert invariant.shape == (b, m, n)
    # invariant [b, m, n] = z @ lambda @ x.T
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
    z_scale = z_invariant.abs().sum(-1, keepdim=True).sqrt().expand(b, m, d)
    # normalize
    z = z / z_scale
    return z


class ScalarMLP(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.group = Lorentz().to(device)
        x_dim = 4
        dim = 128
        self.net = nn.Sequential(
            nn.Linear(16, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, 1)
        ).to(device)
        self.x_dim = x_dim
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                m.bias.data.zero_()

    def _invariant(self, x):
        # [b, n, d] -> [b, 1]
        b, n, _ = x.shape
        x = orbit_separating_invariant(x).reshape(b, n * n)
        return self.net(x)

    def forward(self, x, sample_size=1):
        assert sample_size == 1, 'ScalarMLP does not support sampling'
        # [b, n * d] -> [b, 1]
        bsize = x.shape[0]
        assert x.shape == (bsize, 16)
        x = x.reshape(bsize, 4, 4)  # [b, n, d]
        output = self._invariant(x)  # [b, 1]
        return output, torch.tensor(0.).to(output)


class Generator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.group = Lorentz().to(device)
        x_dim = 4
        z_dim = 4
        out_dim = 4
        dim = 128
        self.invariant_scale = nn.Parameter(torch.ones(x_dim + z_dim, x_dim, device=device))
        self.invariant_bias = nn.Parameter(torch.zeros(x_dim + z_dim, x_dim, device=device))
        self.net = nn.Sequential(
            nn.Linear((x_dim + z_dim) * (x_dim + z_dim), dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, (x_dim + z_dim) * out_dim)
        ).to(device)
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.out_dim = out_dim
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                m.bias.data.zero_()

    def _invariant(self, x):
        # [b, n, d] -> [b, n, m]
        b, n, _ = x.shape
        x = orbit_separating_invariant(x).reshape(b, n * n)
        return self.net(x).reshape(b, n, self.out_dim)

    def forward(self, x):
        # [b, n * d] -> [b, d, channels=d]
        bsize = x.shape[0]
        x = x.reshape(bsize, 4, 4)  # [b, n, d]
        # sample invariant noise
        invariant = torch.zeros(bsize, self.x_dim + self.z_dim, self.x_dim, device=x.device).uniform_(-1, 1)
        invariant = self.invariant_scale[None, :, :] * invariant + self.invariant_bias[None, :, :]
        # induce equivariant noise
        z_add = sample_equivariant_noise(x, invariant[:, :self.x_dim])  # [b, n, d]
        x = x + z_add  # [b, n, d]
        if self.z_dim > 0:
            z_cat = sample_equivariant_noise(x, invariant[:, self.x_dim:])  # [b, m, d]
            x = torch.cat([x, z_cat], dim=1)  # [b, n + m, d]
        w = self._invariant(x)  # [b, n + m, channels]
        output = torch.einsum('bnd,bno->bdo', x, w)  # [b, d, channels=d]
        # normalize
        invariant = orbit_separating_invariant(output.transpose(1, 2))  # [b, channels=d, channels=d]
        scale = invariant.abs().sum(1, keepdim=True).sqrt().expand_as(output)
        output = output / scale
        # return [b, d, channels=d]
        return output


class Canonicalizer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.group = Lorentz().to(device)
        x_dim = 4
        z_dim = 4
        out_dim = 4
        dim = 128
        self.invariant = nn.Parameter(torch.ones(x_dim + z_dim, x_dim, device=device))
        self.net = nn.Sequential(
            nn.Linear((x_dim + z_dim) * (x_dim + z_dim), dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, (x_dim + z_dim) * out_dim)
        ).to(device)
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.out_dim = out_dim
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                m.bias.data.zero_()

    def _invariant(self, x):
        # [b, n, d] -> [b, n, m]
        b, n, _ = x.shape
        x = orbit_separating_invariant(x).reshape(b, n * n)
        return self.net(x).reshape(b, n, self.out_dim)

    def forward(self, x):
        # [b, n * d] -> [b, d, channels=d]
        bsize = x.shape[0]
        assert x.shape == (bsize, 16)
        x = x.reshape(bsize, 4, 4)  # [b, n, d]
        # invariant feature
        invariant = self.invariant[None, :, :].expand(bsize, self.x_dim + self.z_dim, self.x_dim)
        # induce equivariant feature
        z_add = sample_equivariant_noise(x, invariant[:, :self.x_dim])  # [b, n, d]
        x = x + z_add  # [b, n, d]
        if self.z_dim > 0:
            z_cat = sample_equivariant_noise(x, invariant[:, self.x_dim:])  # [b, m, d]
            x = torch.cat([x, z_cat], dim=1)  # [b, n + m, d]
        w = self._invariant(x)  # [b, n + m, channels]
        output = torch.einsum('bnd,bno->bdo', x, w)  # [b, d, channels=d]
        # normalize
        invariant = orbit_separating_invariant(output.transpose(1, 2))  # [b, channels=d, channels=d]
        scale = invariant.abs().sum(1, keepdim=True).sqrt().expand_as(output)
        output = output / scale
        # return [b, d, channels=d]
        return output


class SymmetrizedMLP(nn.Module):
    def __init__(self, device, use_symmetrization=False, use_aug=False):
        super().__init__()
        assert not (use_symmetrization and use_aug)
        self.group = Lorentz().to(device)
        self.use_symmetrization = use_symmetrization
        self.use_aug = use_aug
        self.generator = Generator(device)
        dim = 128
        self.backbone = nn.Sequential(
            nn.Linear(16, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, 1)
        ).to(device)
        # initilize biases to zero and batchnorm to identity
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                m.bias.data.zero_()
        self.aug_gs = self.group.samples(10000).to(device)  # [10000, d, d]

    def samples(self, x):
        # [b, n * d] -> [b, d, channels=d]
        bsize = x.shape[0]
        if self.use_aug and self.training:
            # random data augmentation
            gs = self.aug_gs[torch.randperm(self.aug_gs.shape[0])[:bsize]]  # [b, d, d]
            return gs, torch.tensor(0.).to(gs)
        if not self.use_symmetrization:
            gs = torch.eye(4, dtype=x.dtype, device=x.device).expand(bsize, 4, 4)
            return gs, torch.tensor(0.).to(gs)
        # equivariant transformation
        gs = self.generator(x)  # [b, d, channels=d]
        # representation loss on g
        target = orbit_separating_invariant(torch.eye(4, dtype=x.dtype, device=x.device))[None, :, :].expand(bsize, 4, 4)
        output = orbit_separating_invariant(gs.transpose(1, 2))  # [b, channels=d, d]
        loss = torch.nn.functional.mse_loss(output, target)
        # representation loss on g^-1x
        x = x.reshape(bsize, 4, 4)  # [b, n, d]
        target = orbit_separating_invariant(x)
        output = orbit_separating_invariant(torch.einsum('bnj,bij->bni', x, lorentz_inverse(gs)))  # [b, n, d]
        loss += torch.nn.functional.mse_loss(output, target)
        # return results
        return gs, loss

    def symmetrized_forward(self, x, gs):
        # [b, n * d], [b, d, d] -> [b, 1]
        bsize = x.shape[0]
        assert x.shape == (bsize, 16)
        x = x.reshape(bsize, 4, 4)  # [b, n, d]
        transformed_x = torch.einsum('bnj,bij->bni', x, lorentz_inverse(gs))  # [b, n, d]
        transformed_x = transformed_x.reshape(bsize, 16)  # [b, n * d]
        output = self.backbone(transformed_x)
        return output

    def forward(self, x, sample_size=1):
        # [b, n * d] -> [b, 1]
        if sample_size > 1:
            x = x[None, ...].repeat(sample_size, 1, 1).reshape(-1, *x.shape[1:])
            gs, loss = self.samples(x)
            output = self.symmetrized_forward(x, gs)
            output = output.reshape(sample_size, -1, *output.shape[1:]).mean(0)
            return output, loss
        gs, loss = self.samples(x)
        output = self.symmetrized_forward(x, gs)
        return output, loss


class CanonicalizedMLP(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.group = Lorentz().to(device)
        self.canonicalizer = Canonicalizer(device)
        dim = 128
        self.backbone = nn.Sequential(
            nn.Linear(16, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, 1)
        ).to(device)
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                m.bias.data.zero_()

    def samples(self, x):
        # [b, n * d] -> [b, d, channels=d]
        bsize = x.shape[0]
        # canonicalizing transformation
        gs = self.canonicalizer(x)  # [b, d, channels=d]
        # representation loss on g
        target = orbit_separating_invariant(torch.eye(4, dtype=x.dtype, device=x.device))[None, :, :].expand(bsize, 4, 4)
        output = orbit_separating_invariant(gs.transpose(1, 2))  # [b, channels=d, d]
        loss = torch.nn.functional.mse_loss(output, target)
        # representation loss on g^-1x
        x = x.reshape(bsize, 4, 4)  # [b, n, d]
        target = orbit_separating_invariant(x)
        output = orbit_separating_invariant(torch.einsum('bnj,bij->bni', x, lorentz_inverse(gs)))  # [b, n, d]
        loss += torch.nn.functional.mse_loss(output, target)
        # return results
        return gs, loss

    def symmetrized_forward(self, x, gs):
        # [b, n * d], [b, d, d] -> [b, 1]
        bsize = x.shape[0]
        x = x.reshape(bsize, 4, 4)  # [b, n, d]
        transformed_x = torch.einsum('bnj,bij->bni', x, lorentz_inverse(gs))  # [b, n, d]
        transformed_x = transformed_x.reshape(bsize, 16)  # [b, n * d]
        output = self.backbone(transformed_x)
        return output

    def forward(self, x, sample_size=1):
        # [b, n * d] -> [b, 1]
        gs, loss = self.samples(x)
        output = self.symmetrized_forward(x, gs)
        return output, loss


class Standardize(nn.Module):
    def __init__(self, model, ds_stats):
        super().__init__()
        assert len(ds_stats) == 4
        self.model = model
        self.ds_stats = ds_stats

    def standardize_input(self, x):
        device = x.device
        muin, sin, _, _ = self.ds_stats
        sin = sin.to(device)
        return (x - muin) / sin

    def unstandardize_input(self, x):
        device = x.device
        muin, sin, _, _ = self.ds_stats
        sin = sin.to(device)
        return sin * x + muin

    def unstandardize_output(self, output):
        device = output.device
        _, _, muout, sout = self.ds_stats
        muout, sout = muout.to(device), sout.to(device)
        return sout * output + muout

    def forward(self, x, sample_size=1):
        x = self.standardize_input(x)
        output, loss = self.model(x, sample_size)
        output = self.unstandardize_output(output)
        return output, loss


def main(args):
    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # configure device
    device = configure_device(args)

    # configure data
    train_loader, val_loader, test_loader, val_lorentz, test_lorentz, train_stats = configure_data(args)

    # configure model and optimizer
    model, optimizer = configure_model(args, train_stats)

    # main loop
    val_best_loss = float('inf')
    test_best_loss = float('inf')
    pbar = tqdm.tqdm(range(args.num_epochs))
    for epoch in pbar:
        # training
        train_task_loss = 0
        train_repr_loss = 0
        cnt = 0
        model.train()
        for x, y in train_loader:
            x = x.to(device)  # [b, n * d]
            y = y.to(device)  # [b, 1]
            bsize = x.shape[0]
            # forward
            target = y  # [b, 1]
            output, repr_loss = model(x, args.sample_size)  # [b, 1]
            task_loss = torch.nn.functional.mse_loss(output, target)
            loss = task_loss + repr_loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # logging
            train_task_loss += task_loss.item() * bsize
            train_repr_loss += repr_loss.item() * bsize
            cnt += bsize
        # logging
        train_task_loss = train_task_loss / cnt
        train_repr_loss = train_repr_loss / cnt

        # validation
        val_task_loss = 0
        val_repr_loss = 0
        cnt = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)  # [b, n * d]
                y = y.to(device)
                bsize = x.shape[0]
                # random Lorentz transformation to detect non-invariant cheating solution
                x = torch.einsum('bnj,bij->bni', x.reshape(bsize, 4, 4), val_lorentz[:bsize]).reshape(bsize, 16)  # [b, n * d]
                # forward
                target = y  # [b, 1]
                output, repr_loss = model(x, args.eval_sample_size)  # [b, 1]
                task_loss = torch.nn.functional.mse_loss(output, target)
                # logging
                val_task_loss += task_loss.item() * bsize
                val_repr_loss += repr_loss.item() * bsize
                cnt += bsize
        # logging
        val_task_loss = val_task_loss / cnt
        val_repr_loss = val_repr_loss / cnt

        # testing if validation loss is the best
        if val_task_loss <= val_best_loss:
            test_task_loss = 0
            test_repr_loss = 0
            cnt = 0
            model.eval()
            with torch.no_grad():
                for x, y, in test_loader:
                    x = x.to(device)  # [b, n * d]
                    y = y.to(device)
                    bsize = x.shape[0]
                    x = torch.einsum('bnj,bij->bni', x.reshape(bsize, 4, 4), test_lorentz[:bsize]).reshape(bsize, 16)  # [b, n * d]
                    # forward
                    target = y  # [b, 1]
                    output, repr_loss = model(x, args.eval_sample_size)  # [b, 1]
                    task_loss = torch.nn.functional.mse_loss(output, target)
                    # logging
                    test_task_loss += task_loss.item() * bsize
                    test_repr_loss += repr_loss.item() * bsize
                    cnt += bsize
            # logging
            test_task_loss = test_task_loss / cnt
            test_repr_loss = test_repr_loss / cnt
            # update best loss
            val_best_loss = val_task_loss
            test_best_loss = test_task_loss

        # logging
        pbar.set_description(f"Epoch [{epoch}/{args.num_epochs}]")
        pbar.set_postfix({
            'loss': f'{train_task_loss:.2e}',
            'repr': f'{train_repr_loss:.2e}',
            'val_loss': f'{val_task_loss:.2e}',
            'val_repr': f'{val_repr_loss:.2e}',
            'val_best': f'{val_best_loss:.2e}',
            'test_best': f'{test_best_loss:.3e}'
        })

    # logging
    print(f"Test loss at best validation loss: {test_best_loss}")


if __name__ == '__main__':
    args_ = get_args()
    main(args_)
