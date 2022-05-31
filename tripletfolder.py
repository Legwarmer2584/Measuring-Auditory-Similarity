# From
# https://github.com/layumi/Person-reID-verification/blob/master/tripletfolder.py

from torchvision import datasets
import torch
import numpy as np
import random
from config import config

class TripletLoader():
    """
    
    """
    def __init__(self, ds) -> None:
        self.ds = ds # Loader contains absolutely every sample and targets
        self.targets = np.asarray([s[1] for s in self.ds]) # Extract a list of every target

    def _get_pos_sample(self, target, index):
        pos_index = np.argwhere(self.targets == target.item())
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = random.randint(0, len(pos_index)-1)
        return self.ds[pos_index[rand]]
    
    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target.item())
        neg_index = neg_index.flatten()
        rand = random.randint(0,len(neg_index)-1)
        k = self.ds[neg_index[rand]]
        return k

    def __getitem__(self, index):
        sample, target = self.ds[index]

        # pos_path, neg_path
        pos_sample = self._get_pos_sample(target, index)[0]
        neg_sample = self._get_neg_sample(target)[0]

        # We want to apply the transform after the batch
        """ if self.transform is not None:
            sample = self.transform(sample)
            pos_sample = self.transform(pos_sample)
            neg_sample = self.transform(neg_sample) """

        #target = torch.tensor(target) # <- Is already a tensor. Not needed

        return sample, target, pos_sample, neg_sample
    
    def __len__(self):
        return len(self.ds)

class TripletFolder(datasets.ImageFolder):

    def __init__(self, root, transform=None):
        super(TripletFolder, self).__init__(root, transform)
        self.targets = np.asarray([s[1] for s in self.samples]) # The classes

    def _get_pos_sample(self, target, index):
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = random.randint(0, len(pos_index)-1)
        return self.samples[pos_index[rand]]

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0,len(neg_index)-1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        path, target = self.samples[index]

        # pos_path, neg_path
        pos_path = self._get_pos_sample(target, index)
        neg_path = self._get_neg_sample(target)

        sample = self.loader(path)
        pos = self.loader(pos_path[0])
        neg = self.loader(neg_path[0])

        if self.transform is not None:
            sample = self.transform(sample)
            pos = self.transform(pos)
            neg = self.transform(neg)

        target = torch.tensor(target)

        return sample, target, pos, neg