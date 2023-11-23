#import sklearn
import scipy
#import skimage
import pandas
import numpy as np
from PIL import Image
#import bokeh
from copy import deepcopy

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
#%matplotlib inline
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg', 'pdf') # For export
import matplotlib
#matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models

## Torchvision
import torchvision
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Import tensorboard
#%load_ext tensorboard

from tensorboard.plugins import projector

import cv2
import pathlib
import os
import datetime
#import tensorflow as tf

from os import listdir, walk
from os.path import isfile, join


from os import listdir, walk
from os.path import isfile, join

import wandb
import random

data_path_train = "/gpfs/data/fs71186/kadic/Herbarium_2022/train_images"
data_path_test = "/gpfs/data/fs71186/kadic/Herbarium_2022/test_images"

ground_truths = "/gpfs/data/fs71186/kadic/Herbarium_2022/train_metadata.json"

train_image_files = [join(dirpath,f) for (dirpath, dirnames, filenames) in walk(data_path_train) for f in filenames] 

print(len(train_image_files))

test_image_files = [join(dirpath,f) for (dirpath, dirnames, filenames) in walk(data_path_test) for f in filenames] 

print(len(test_image_files))

train_image_files = sorted(train_image_files)
test_image_files = sorted(test_image_files)

import json
 
# Opening JSON file
f = open(ground_truths)
 
# returns JSON object as
# a dictionary
ground_truth_data = json.load(f)

print(ground_truth_data.keys())
print(len(ground_truth_data["annotations"]))
print(ground_truth_data["annotations"][0])
print(train_image_files[0])

print(len(ground_truth_data["categories"]))
print(len(ground_truth_data["genera"]))
      
gt_annot = ground_truth_data["annotations"]
# Iterating through the json
# list 
#Closing file

f.close()

train_data = []
test_data = []

for i, img in enumerate(train_image_files):
    if(i % 100000 == 0):
        #print(i)
        print(img)
        print(gt_annot[i]['image_id'])
    train_data.append((img, gt_annot[i]['category_id']))

print(len(train_image_files))
print(len(train_data))

labels = []
label_count = {}
for img, annot in train_data:
    if annot not in labels:
        labels.append(annot)
        label_count[str(annot)] = 1
    else:
        label_count[str(annot)] = int(label_count[str(annot)]) + 1
        
#print(len(labels))
#print(label_count)
sorted_count = dict(sorted(label_count.items(), key=lambda item: item[1]))

#print(list(sorted_count.items())[:200])

print(len(list(sorted_count.items())))
biggest_categories = list(sorted_count.items())[15301:]
#print(list(sorted_count.items())[12333:])

print(len(biggest_categories))
cat_sum = 0
for cat in biggest_categories:
    cat_sum += cat[1]
print(cat_sum)

dict_keys_back = {}
for i, cat in enumerate(biggest_categories):
    dict_keys_back[str(cat[0])] = i
#print(dict_keys_back)

reduced_train_data = []
for img, annot in train_data:
    for key, val in biggest_categories:
        if str(annot) == str(key):
            reduced_train_data.append((img, annot))

print(len(reduced_train_data))

reduced_train_data_subbed = []
for img, annot in train_data:
    for key, val in biggest_categories:
        if str(annot) == str(key):
            reduced_train_data_subbed.append((img, dict_keys_back[str(annot)]))

print(len(reduced_train_data_subbed))

class ImageDataset(Dataset):
    def __init__(self, paths,transform):
        self.paths = [i[0] for i in paths]
        self.transform = transform
        self.target_paths = [i[1] for i in paths]
        
    def __len__(self):
        return len(self.paths)
    

    def __getitem__(self, index):
        image_path = self.paths[index]
        image_l = Image.open(image_path)
        image = image_l.convert('RGB')
        image_tensor = image
        if self.transform:
            image_tensor = self.transform(image)
        
        target = self.target_paths[index]
        
        return (image_tensor, target)
    

#tr_files = reduced_train_data_subbed[0:12772] #train_data[0:543991]
#ts_files = reduced_train_data_subbed[12772:15965] #train_data[543991:]

cpy = reduced_train_data_subbed

import random

random.shuffle(cpy)

tr_files = cpy[0:12780] #train_data[0:543991]
ts_files = cpy[12780:15974] #train_data[543991:]

print(len(tr_files))

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = data_path_train
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/contrastive_models"
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count()

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

class ResNet(pl.LightningModule):

    def __init__(self, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs*0.7),
                                                                  int(self.hparams.max_epochs*0.9)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode='train'):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + '_loss', loss)
        self.log(mode + '_acc', acc)
        
        wandb.log({"acc": acc, "loss": loss})
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')
        

print("256 + 256 + 1.0, btch_128 ")

train_transforms = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                       transforms.Resize((256, 256)),
                                       #transforms.RandomResizedCrop(size=256, scale=(1.0, 1.0)),
                                       transforms.RandomGrayscale(p=0.2),
                                       #transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))
                                       ])
train_img_data = ImageDataset(tr_files, transform = train_transforms)

img_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])

test_img_data = ImageDataset(ts_files, transform = img_transforms)

print("Number of training examples:", len(train_img_data))
print("Number of test examples:", len(test_img_data))

print(type(train_img_data[0]))


def train_resnet(batch_size, max_epochs=100, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "ResNet"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")],
                         check_val_every_n_epoch=2)
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(train_img_data, batch_size=batch_size, shuffle=True,
                                   drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = data.DataLoader(test_img_data, batch_size=batch_size, shuffle=False,
                                  drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ResNet_4.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        model = ResNet.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42) # To be reproducable
        model = ResNet(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = ResNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    val_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": val_result[0]["test_acc"]}

    return model, result


wandb.init(
    # set the wandb project where this run will be logged
    project="ResNet-Baseline",
    name= "200ep-resnet-noflip-norecrop-256-15k",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-3,
    "architecture": "ResNet-18",
    "dataset": "Subset-200",
    "epochs": 200,
    }
)


resnet_model, resnet_result = train_resnet(batch_size=128,
                                           num_classes=200,
                                           lr=1e-3,
                                           weight_decay=2e-4,
                                           max_epochs=200)

print("NO RECROP, NO FLIP, 256X256, 200 largest categories, 200 epochs")
for i in range(20):
    print("###########")
print(f"Accuracy on training set: {100*resnet_result['train']:4.2f}%")
print(f"Accuracy on test set: {100*resnet_result['test']:4.2f}%")