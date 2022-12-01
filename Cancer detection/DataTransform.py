###################################
########                 ##########
#######   BY CapLabs    ###########     
########                 ##########
###################################

# Importing Libraries

import torch
import torchvision
from torchvision import transforms, datasets
from pathlib import Path
from typing import List, Dict
from BreastDataset import BreastDataset
#from BreastDataset.BreastDataset import splitting
from torch.utils.data import DataLoader
import os
from PIL import Image
import png

# Select device based on GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class DataLoad():
    """
    Python class for loading data fro training
    """
    def __init__(
            self,
            in_shape: int = 224,
            mean: List = [0.5, 0.5, 0.5],
            std: List = [0.5, 0.5, 0.5], ):

        """
        data_dir : main data folder
        in_shape : input images shape
        mean : mean value for normalization
        std : standart deviation values for normalization
        """
        
        self.in_shape = in_shape
        self.mean = mean
        self.std = std
        
        self.data_dir = "./BreastTrainingData"

    def filenames(self, output_dir, image_arrays, labels, subfolder):
        """
        output_dir : main data folder
        image_arrays : training, testing or val data
        subfolder : subfolder directory (train, test, val)
        labels : 0, 1 or 2 labels
        """
        for idx, image in enumerate(image_arrays):
            image=image.astype(float)/image.max()
            image=image*65535
            image=image.astype("uint16")
            image = png.from_array(image,mode="L;16")

            if labels[idx] == 0:
                image.save(os.path.join(output_dir, f'{subfolder}', f"{0}", f"{idx}.png"))
            elif labels[idx] == 1:
                image.save(os.path.join(output_dir, f'{subfolder}', f"{1}", f"{idx}.png"))
            else:
                image.save(os.path.join(output_dir, f'{subfolder}', f"{2}", f"{idx}.png"))
           

# Data transformation and normalization
    def DataTransforms(self,):

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.in_shape),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)]),

            'val': transforms.Compose([
                transforms.CenterCrop(self.in_shape),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)]),

            'test': transforms.Compose([
                transforms.CenterCrop(self.in_shape),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)]),
            }
        
        Bdata = BreastDataset()
        data_list, labels_list = Bdata.splitting()

        self.filenames(self.data_dir, data_list[0], labels_list[0], 'train')
        self.filenames(self.data_dir, data_list[1], labels_list[1], 'val')
        self.filenames(self.data_dir, data_list[2], labels_list[2], 'test')
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x])
                          for x in ['train', 'val', 'test']}

        dataloader = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                      for x in ['train', 'val', 'test']}        
        
        print(dataloader.keys())
        return dataloader



