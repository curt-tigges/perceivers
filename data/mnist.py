# To get the layers and losses for our model
import pytorch_lightning as pl


# To get MNIST data and transforms
from torchvision import datasets, transforms

# To get random_split to split training
# data into training and validation data
# and DataLoader to create dataloaders for train,
# valid and test data to be returned
# by our data module
from torch.utils.data import random_split, DataLoader


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, download_dir, batch_size, num_workers):
        super().__init__()

        # Directory to store MNIST Data
        self.download_dir = download_dir

        # Defining batch size of our data
        self.batch_size = batch_size

        self.num_workers = num_workers

        # Defining transforms to be applied on the data
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):

        # Downloading our data
        datasets.MNIST(self.download_dir, train=True, download=True)

        datasets.MNIST(self.download_dir, train=False, download=True)

    def setup(self, stage=None):

        # Loading our data after applying the transforms
        data = datasets.MNIST(self.download_dir, train=True, transform=self.transform)

        self.train_data, self.valid_data = random_split(data, [55000, 5000])

        self.test_data = datasets.MNIST(
            self.download_dir, train=False, transform=self.transform
        )

    def train_dataloader(self):

        # Generating train_dataloader
        return DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):

        # Generating val_dataloader
        return DataLoader(
            self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):

        # Generating test_dataloader
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )
