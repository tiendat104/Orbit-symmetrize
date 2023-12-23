"""Train a regression model on a dataset."""
import logging
import torch
from torch.utils.data import DataLoader
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed
from oil.datasetup.datasets import split_dataset

from emlp_pytorch.trainer.model_trainer import RegressorPlus
from emlp_pytorch.trainer.utils import LoaderTo
from emlp_pytorch.nn import MLP, EMLP, Standardize
from emlp_pytorch.interface import Interface, GroupAugmentation
from emlp_pytorch.datasets import Inertia


log_levels = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}
device = 'cuda'
torch.set_default_dtype(torch.float64)


def makeTrainer(*, emlp=True, ndata=10000+2000, seed=2021, lr=3e-3, bs=500,
                aug=False, n_samples=1, test_aug=False, test_n_samples=1, reg=None,
                trainable_aug=False):
    """ Make a trainer for a regression problem on a dataset. """
    logging.getLogger().setLevel(log_levels['info'])
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed), FixedPytorchSeed(seed):
        base_dataset = Inertia(ndata)
        datasets = split_dataset(base_dataset, splits={
                                 'train': -1, 'val': 1000, 'test': 1000})
    model = (EMLP if emlp else MLP)(rep_in=base_dataset.rep_in,
                                    rep_out=base_dataset.rep_out,
                                    group=base_dataset.symmetry,
                                    ch=384, num_layers=3, device=device)
    if aug:
        if trainable_aug:
            model = Interface(
                model, base_dataset.rep_in, base_dataset.rep_out, base_dataset.symmetry,
                n_samples, test_aug, test_n_samples, device)
        else:
            model = GroupAugmentation(
                model, base_dataset.rep_in, base_dataset.rep_out, base_dataset.symmetry,
                n_samples, test_aug, test_n_samples)
    model = Standardize(model, datasets['train'].stats)
    dataloaders = {k: LoaderTo(DataLoader(v, batch_size=min(bs, len(v)),
                                          shuffle=(k == 'train'),
                                          num_workers=0,
                                          pin_memory=False))
                   for k, v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']

    return RegressorPlus(model, reg, dataloaders, torch.optim.Adam, lr_sched=lambda _: lr,
                         log_dir=None, log_args={'minPeriod': .02, 'timeFrac': .75},
                         early_stop_metric='val_MSE')


if __name__ == "__main__":
    n_samples = 1
    total_bs = 500
    test_n_samples = 10
    trainer = makeTrainer(emlp=False, bs=total_bs//n_samples,
                          aug=True, n_samples=n_samples,
                          test_aug=True, test_n_samples=test_n_samples,
                          reg=None, trainable_aug=True)
    trainer.train(num_epochs=1000)
