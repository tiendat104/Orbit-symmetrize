import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, \
    verify_str_arg

from oil.utils.utils import LoaderTo, cosLr, islice
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from oil.utils.parallel import try_multigpu_parallelize
from oil.model_trainers.classifier import Classifier
from functools import partial
from torch.optim import Adam
from oil.tuning.args import argupdated_config

from PIL import Image
import math
import numpy as np
import copy
import statistics 
import os

class MnistRotDataset(VisionDataset):
    """ Official RotMNIST dataset."""
    ignored_index = -100
    class_weights = None
    balanced = True
    stratify = True
    means = (0.130,)
    stds = (0.297,)
    num_targets=10
    resources = ["http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"]
    training_file = 'mnist_all_rotation_normalized_float_train_valid.amat'
    test_file = 'mnist_all_rotation_normalized_float_test.amat'
    def __init__(self,root, mode="train", transform=None,download=True, num_train=10000):
        super().__init__(root,transform=transform)
        assert mode in ["train", "val", "test"]
        self.mode = mode
        if download:
            self.download()
            
        if self.mode == "train" or self.mode == "val":
            file=os.path.join(self.raw_folder, self.training_file)
        elif self.mode == "test":
            file=os.path.join(self.raw_folder, self.test_file)
        
        self.transform = transforms.ToTensor()

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32) # [0, 1]
        self.labels = data[:, -1].astype(np.int64)
        
        # Split train and val
        if self.mode == "train":
            self.images = self.images[:num_train]
            self.labels = self.labels[:num_train]
        elif self.mode == "val":
            self.images = self.images[num_train:]
            self.labels = self.labels[num_train:]
        else:
            pass
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        # image.min = 0, image.max = 0.991809
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)
        return image, label
    def _check_exists(self):
        return (os.path.exists(os.path.join(self.raw_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.raw_folder,
                                            self.test_file)))
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')
    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder,exist_ok=True)
        os.makedirs(self.processed_folder,exist_ok=True)

        # download files
        for url in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=None)
        print('Downloaded!')

    def __len__(self):
        return len(self.labels)
    def default_aug_layers(self):
        return RandomRotateTranslate(0)# no translation