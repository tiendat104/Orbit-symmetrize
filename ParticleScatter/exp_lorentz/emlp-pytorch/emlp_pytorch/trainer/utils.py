""" Utilities for the trainer. """
import functools
import torch
from oil.utils.utils import imap


def minibatch_to(mb):
    try:
        if isinstance(mb, torch.Tensor):  # TODO: send to device
            return mb
        return mb
    except AttributeError:
        if isinstance(mb, dict):
            return type(mb)(((k, minibatch_to(v)) for k, v in mb.items()))
        else:
            return type(mb)(minibatch_to(elem) for elem in mb)


def LoaderTo(loader):
    return imap(functools.partial(minibatch_to), loader)
