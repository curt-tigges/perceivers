import os
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset, DataLoader
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


class CIFAR100DataModule(pl.LightningDataModule):
    '''Simple data module that downloads CIFAR-100 and inserts into dataloaders

    This datamodule will download the dataset if it is not present, transform it,
    split it into test, train and validation datasets, and create a dataloader
    for each.

    Args:
        batch_size (int): Desired batch size for dataloaders.
        num_workers (int): Number of CPU threads to use.
        classes (int): Number of classes (100 in CIFAR-100)
        data_dir (str): Directory into which data is located or is to be downloaded
    '''
    def __init__(self, batch_size=128, num_workers=0, classes=100, data_dir=None) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = classes
        self.data_dir = (os.getcwd() if data_dir is None else data_dir)

    def prepare_data(self) -> None:
        '''Downloads data if it is not located in data_dir.

        Args: None
        
        Returns: None
        '''
        CIFAR100(root=self.data_dir, train=True, download=True)
        CIFAR100(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        '''Applies desired transforms and creates datasets
        
        Args:
            stage (str): Required by Pytorch Lightning. Unused here.

        '''
        train_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=[32,32]),
            transforms.ToTensor(),
            cifar10_normalization()]
        )

        test_transforms = transforms.Compose(
            [transforms.ToTensor(),
            cifar10_normalization()]
        )

        cifar100_train = CIFAR100(
            root=self.data_dir, train=True, download=True, 
            transform=train_transforms)
        cifar100_val = CIFAR100(
            root=self.data_dir, train=True, download=True, 
            transform=test_transforms)
        
        # we can use this trick to easily split off the val set from the train set
        pl.seed_everything(42)
        self.train_set, _ = random_split(cifar100_train, [45000, 5000])
        pl.seed_everything(42)
        _, self.val_set = random_split(cifar100_val, [45000, 5000])

        self.test_set = CIFAR100(
            root=self.data_dir, train=False, download=True, 
            transform=test_transforms)

    def train_dataloader(self) -> DataLoader:
        '''Returns train dataloader
        
        Returns:
            Dataloader for training set
        '''
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        '''Returns validation dataloader
        
        Returns:
            Dataloader for validation set
        '''
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        '''Returns test dataloader
        
        Returns:
            Dataloader for test set
        '''
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)