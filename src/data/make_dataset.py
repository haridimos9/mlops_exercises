# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

####################3
# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import torch
from torchvision.transforms import ToTensor, Normalize


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(_name_)
    logger.info('making final data set from raw data')

    # Get the training and test images and labels as numpy arrays
    train_images, train_labels = read_data(input_filepath)
    test_images, test_labels = read_data(input_filepath, 'npz', False)


    # Convert them in pytorch tensors
    train_images = torch.from_numpy(train_images)
    train_labels = torch.from_numpy(train_labels)

    test_images = torch.from_numpy(test_images)
    test_labels = torch.from_numpy(test_labels)

    # Normalize the images
    img_mean = 0.0
    img_std = 1.0

    transforms = torch.nn.Sequential(
        ToTensor(),
        Normalize(img_mean,img_std)
    )

    #train_images = transforms(train_images.reshape(train_images.shape[0], train_images.shape[1],1))
    #test_images = transforms(test_images.reshape(test_images.shape[0], test_images.shape[1],1))

    torch.save(train_images, output_filepath+'/train_images.pt')
    torch.save(train_labels, output_filepath+'/train_labels.pt')
    torch.save(test_images, output_filepath+'/test_images.pt')
    torch.save(test_labels, output_filepath+'/test_labels.pt')




def read_data(filepath, suffix='npz', training_dataset=True):
    
    # Iterate through the files of the input dir for the desired files
    # The suffix is .npz
    type_str = 'train'
    if not training_dataset:
        type_str = 'test'
    
    first_pass = True
    data_len = 0
    for file_ in Path(filepath).glob(f'{type_str}*.{suffix}'):
        with np.load(file_) as f:
            data_len += len(f['labels'])
            if first_pass:
                height, width = f['images'][0].shape
            first_pass = False

    # Preallocate a numpy tensor for the images and labels
    images = np.zeros([ data_len, height, width ])
    labels = np.zeros(data_len)

    prev_idx = 0
    for file_ in Path(filepath).glob(f'{type_str}*.{suffix}'):
        with np.load(file_) as f:
            current_labels = f['labels']
            current_idx = prev_idx + len(current_labels)
            
            labels[prev_idx:current_idx] = current_labels[:]
            
            images[prev_idx:current_idx,:,:] = f['images']

            prev_idx = current_idx
    
    # # Sanity check and not useless to know
    # print(f"{type_str} labels shape: {labels.shape}")
    # print(f"{type_str} images shape: {images.shape}")
    return images, labels



if _name_ == '_main_':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(_file_).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

######################
# -*- coding: utf-8 -*-
import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import torch
import wget
from torch import Tensor
from torch.utils.data import Dataset


class CorruptMnist(Dataset):
    def __init__(self, train: bool, in_folder: str = "", out_folder: str = "") -> None:
        super().__init__()

        self.train = train
        self.in_folder = in_folder
        self.out_folder = out_folder

        if self.out_folder:  # try loading from proprocessed
            try:
                self.load_preprocessed()
                print("Loaded from pre-processed files")
                return
            except ValueError:  # not created yet, we create instead
                pass
    # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

        if self.train:
            content = []
            for i in range(5):
                content.append(np.load(f"{in_folder}/train_{i}.npz", allow_pickle=True))
            data = torch.tensor(np.concatenate([c["images"] for c in content])).reshape(
                -1, 1, 28, 28
            )
            targets = torch.tensor(np.concatenate([c["labels"] for c in content]))
        else:
            content = np.load(f"{in_folder}/test.npz", allow_pickle=True)
            data = torch.tensor(content["images"]).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content["labels"])

        self.data = data
        self.targets = targets

        if self.out_folder:
            self.save_preprocessed()

    def save_preprocessed(self) -> None:
        split = "train" if self.train else "test"
        torch.save([self.data, self.targets], f"{self.out_folder}/{split}_processed.pt")

    def load_preprocessed(self) -> None:
        split = "train" if self.train else "test"
        try:
            self.data, self.targets = torch.load(f"{self.out_folder}/{split}_processed.pt")
        except:
            raise ValueError("No preprocessed files found")


    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx].float(), self.targets[idx]


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train = CorruptMnist(train=True, in_folder=input_filepath, out_folder=output_filepath)
    train.save_preprocessed()

    test = CorruptMnist(train=False, in_folder=input_filepath, out_folder=output_filepath)
    test.save_preprocessed()

    print(train.data.shape)
    print(train.targets.shape)
    print(test.data.shape)
    print(test.targets.shape)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()