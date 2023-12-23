""" Trainer subclass. """
import torch
import torch.nn as nn  # pylint: disable=R0402:consider-using-from-import
from .trainer import Trainer


def cross_entropy(logprobs, targets):
    """ Cross-entropy loss. Assumes logprobs is (N, C) and targets is (N,) """
    ll = torch.take_along_dim(logprobs, targets.unsqueeze(1), 1)
    ce = -torch.mean(ll)
    return ce


class Classifier(Trainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """

    def loss(self, minibatch):
        """ Standard cross-entropy loss """  # TODO: support class weights
        x, y = minibatch
        device = next(self.model.model.parameters()).device
        x = x.to(device)
        y = y.to(device)
        logits = self.model(x)
        logp = nn.LogSoftmax(logits)
        return cross_entropy(logp, y)

    def metrics(self, loader):
        device = next(self.model.model.parameters()).device
        def acc(mb):
            return torch.mean(torch.argmax(
                self.model(mb[0].to(device)), axis=-1) == mb[1].to(device))
        return {'Acc': self.evalAverageMetrics(loader, acc)}


class Regressor(Trainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        x, y = minibatch
        device = next(self.model.model.parameters()).device
        x = x.to(device)
        y = y.to(device)
        mse = torch.mean((self.model(x)-y)**2)
        return mse

    def metrics(self, loader):
        device = next(self.model.model.parameters()).device
        def mse(mb):
            return torch.mean((self.model(mb[0].to(device))-mb[1].to(device))**2)
        return {'MSE': self.evalAverageMetrics(loader, mse)}
