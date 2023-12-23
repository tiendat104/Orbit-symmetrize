""" Trainer class. Implements training loop, logging, checkpointing, etc. """
from functools import partial
from itertools import islice
import torch
from .classifier import Regressor, Classifier


def rel_err(a, b):
    """ Relative error between two tensors. """
    return torch.sqrt(((a-b)**2).mean())/(torch.sqrt((a**2).mean())+torch.sqrt((b**2).mean()))


def scale_adjusted_rel_err(a, b, g):
    """ Relative error between two tensors. """
    return torch.sqrt(((a-b)**2).mean())/ \
        (torch.sqrt((a**2).mean())+
         torch.sqrt((b**2).mean())+
         torch.abs(g-torch.eye(g.size(-1), device=a.device)).mean())


def equivariance_err(model, mb, group=None):
    """ Equivariance error. """
    x, _ = mb
    device = next(model.model.parameters()).device
    x = x.to(device)
    group = model.model.G if group is None else group
    gs = group.samples(x.size(0))
    rho_gin = torch.vmap(model.model.rep_in.rho_dense)(gs)
    rho_gout = torch.vmap(model.model.rep_out.rho_dense)(gs)
    y1 = model((rho_gin@x[..., None])[..., 0])
    y2 = (rho_gout@model(x)[..., None])[..., 0]
    return scale_adjusted_rel_err(y1, y2, gs).detach().item()


def equivariance_err_reg(model, mb, group=None):
    """ Equivariance error. """
    x, _ = mb
    device = next(model.model.parameters()).device
    x = x.to(device)
    group = model.model.G if group is None else group
    gs = group.samples(x.size(0))
    rho_gin = torch.vmap(model.model.rep_in.rho_dense)(gs)
    rho_gout = torch.vmap(model.model.rep_out.rho_dense)(gs)
    y1 = model((rho_gin@x[..., None])[..., 0])
    y2 = (rho_gout@model(x)[..., None])[..., 0]
    return scale_adjusted_rel_err(y1, y2, gs)


class RegressorPlus(Regressor):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """

    def __init__(self, model, reg, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.model = model
        self.reg = reg

    def get_device(self):
        """ Get device of model parameters. """
        return next(self.model.model.parameters()).device

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        device = self.get_device()
        x, y = minibatch
        x = x.to(device)
        y = y.to(device)
        mse = torch.mean((self.model(x)-y)**2)
        if self.reg is not None:
            reg_loss = self.reg * equivariance_err_reg(self.model, minibatch)
            mse = mse + reg_loss
        return mse

    @torch.no_grad()
    def metrics(self, loader):
        device = self.get_device()
        def mse(mb):
            x, y = mb[0].to(device), mb[1].to(device)
            return torch.mean((self.model(x)-y)**2).detach().item()
        return {'MSE': self.evalAverageMetrics(loader, mse)}

    @torch.no_grad()
    def logStuff(self, step, minibatch=None):
        metrics = {}
        metrics['test_equivar_err'] = self.evalAverageMetrics(
            islice(self.dataloaders['test'], 0, None, 5),
            partial(equivariance_err, self.model))  # subsample by 5x so it doesn't take too long
        self.logger.add_scalars('metrics', metrics, step)
        super().logStuff(step, minibatch)


class ClassifierPlus(Classifier):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.model = model

    def logStuff(self, step, minibatch=None):
        metrics = {}
        metrics['test_equivar_err'] = self.evalAverageMetrics(
            islice(self.dataloaders['test'], 0, None, 5),
            partial(equivariance_err, self.model))  # subsample by 5x so it doesn't take too long
        self.logger.add_scalars('metrics', metrics, step)
        super().logStuff(step, minibatch)
