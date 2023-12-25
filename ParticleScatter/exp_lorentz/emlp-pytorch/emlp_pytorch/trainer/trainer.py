""" Base trainer """
import copy
import torch
from torch import nn
from oil.logging.lazyLogger import LazyLogger
from oil.utils.mytqdm import tqdm
from oil.tuning.study import guess_metric_sign


class Trainer(nn.Module):
    """ Base trainer """

    def __init__(self, model, dataloaders, optim=torch.optim.Adam, lr_sched=lambda e: 1,
                 log_dir=None, log_suffix='', log_args={}, early_stop_metric=None):
        super().__init__()
        # Setup model, optimizer, and dataloaders
        self.model = model

        self.optimizer = optim(model.parameters())
        self.lr_sched = lr_sched
        self.dataloaders = dataloaders  # A dictionary of dataloaders
        self.epoch = 0

        self.logger = LazyLogger(log_dir, log_suffix, **log_args)
        self.hypers = {}
        # copy.deepcopy(self.state_dict()) # TODO: fix model saving
        self.ckpt = None
        self.early_stop_metric = early_stop_metric

    def metrics(self, loader):
        """ Returns a dictionary of metrics for a given loader """
        return {}

    def loss(self, minibatch):
        """ Returns the loss for a given minibatch """
        raise NotImplementedError

    def train_to(self, final_epoch=100):
        """ Trains to a given epoch """
        assert final_epoch >= self.epoch, "trying to train less than already trained"
        self.train(final_epoch-self.epoch)

    def train(self, num_epochs=100):
        """ The main training loop"""
        start_epoch = self.epoch
        steps_per_epoch = len(self.dataloaders['train'])
        step = 0
        for self.epoch in tqdm(range(start_epoch, start_epoch + num_epochs), desc='train'):
            for i, minibatch in enumerate(self.dataloaders['train']):
                step = i + self.epoch*steps_per_epoch
                self.step(self.epoch+i/steps_per_epoch, minibatch)
                with self.logger as do_log:
                    if do_log:
                        self.model.eval()
                        self.logStuff(step, minibatch)
                        self.model.train()
        self.epoch += 1
        self.model.eval()
        self.logStuff(step)
        self.model.train()

    def step(self, epoch, minibatch):
        """ Training step """
        self.optimizer.zero_grad()
        loss = self.loss(minibatch)
        loss.backward()
        self.optimizer.step()  # TODO: lr scheduling
        return loss

    def logStuff(self, step, minibatch=None):
        """ Logs metrics and model data """
        metrics = {}
        for loader_name, dloader in self.dataloaders.items():  # Ignore metrics on train
            if loader_name == 'train' or len(dloader) == 0 or loader_name[0] == '_':
                continue
            for metric_name, metric_value in self.metrics(dloader).items():
                metrics[loader_name+'_'+metric_name] = metric_value
        self.logger.add_scalars('metrics', metrics, step)
        self.logger.report()
        # update the best checkpoint
        if self.early_stop_metric is not None:
            maximize = guess_metric_sign(self.early_stop_metric)
            sign = 2*maximize-1
            best = (sign*self.logger.scalar_frame[self.early_stop_metric].values).max()
            current = sign * self.logger.scalar_frame[self.early_stop_metric].iloc[-1]
            if current >= best:
                self.ckpt = copy.deepcopy(self.state_dict())
        else:
            self.ckpt = copy.deepcopy(self.state_dict())

    def evalAverageMetrics(self, loader, metrics):
        """ Returns the average of metrics over a dataloader """
        num_total, loss_totals = 0, 0
        for minibatch in loader:
            try:
                mb_size = loader.batch_size
            except AttributeError:
                mb_size = 1
            loss_totals += mb_size*metrics(minibatch)
            num_total += mb_size
        if num_total == 0:
            raise KeyError("dataloader is empty")
        return loss_totals/num_total

    def state_dict(self):
        """ Returns a dictionary of the state of the trainer """
        # TODO: handle saving and loading state
        state = {
            'outcome': self.logger.scalar_frame[-1:],
            'epoch': self.epoch,
            # 'model_state':self.model.state_dict(),
            # 'optim_state':self.optimizer.state_dict(),
            # 'logger_state':self.logger.state_dict(),
        }
        return state

    # def load_state_dict(self,state):
    #     self.epoch = state['epoch']
    #     self.model.load_state_dict(state['model_state'])
    #     self.optimizer.load_state_dict(state['optim_state'])
    #     self.logger.load_state_dict(state['logger_state'])

    # def load_checkpoint(self,path=None):
    #     """ Loads the checkpoint from path, if None gets the highest epoch checkpoint"""
    #     if not path:
    #         chkpts = glob.glob(os.path.join(self.logger.log_dirr,'checkpoints/c*.state'))
    #         path = natsorted(chkpts)[-1] # get most recent checkpoint
    #         print(f"loading checkpoint {path}")
    #     with open(path,'rb') as f:
    #         self.load_state_dict(dill.load(f))

    # def save_checkpoint(self):
    #     return self.logger.save_object(self.ckpt,suffix=f'checkpoints/c{self.epoch}.state')
