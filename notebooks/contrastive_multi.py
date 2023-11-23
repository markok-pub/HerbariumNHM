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

import wandb
import random


data_path_train = "/gpfs/data/fs71186/kadic/Herbarium_2022/train_images"
data_path_test = "/gpfs/data/fs71186/kadic/Herbarium_2022/test_images"

from os import listdir, walk
from os.path import isfile, join

train_image_files = [join(dirpath,f) for (dirpath, dirnames, filenames) in walk(data_path_train) for f in filenames] 

print(len(train_image_files))

test_image_files = [join(dirpath,f) for (dirpath, dirnames, filenames) in walk(data_path_test) for f in filenames] 

print(len(test_image_files))

################################################################################
################################################################################
# MULTIPLE AUGMENTATIONS FOR LESS REPRESENTED SPECIMENS - MIN 60.
CONST_ENLARGE = 60
ground_truths = "/gpfs/data/fs71186/kadic/Herbarium_2022/train_metadata.json"

train_image_files = sorted(train_image_files)
test_image_files = sorted(test_image_files)

import json
# Opening JSON file
f = open(ground_truths)
ground_truth_data = json.load(f)
gt_annot = ground_truth_data["annotations"]
f.close()

train_data = []
test_data = []

for i, img in enumerate(train_image_files):
    train_data.append((img, gt_annot[i]['category_id']))

labels = []
label_count = {}
for img, annot in train_data:
    if annot not in labels:
        labels.append(annot)
        label_count[str(annot)] = 1
    else:
        label_count[str(annot)] = int(label_count[str(annot)]) + 1
        
sorted_count = dict(sorted(label_count.items(), key=lambda item: item[1]))

enlarge_dict = {}
for cat,cnt in list(sorted_count.items()):
    if(int(cnt) < CONST_ENLARGE):
        enlarge_dict[cat] = cnt

needed_categories = enlarge_dict
print(len(needed_categories))

cat_sum = 0
for key in needed_categories:
    cat_sum += needed_categories[key]
print(cat_sum)

added_train_data = []
used = []
for img, annot in train_data:
    if (str(annot) in needed_categories) and (annot not in used):
        used.append(annot)
        for i in range(CONST_ENLARGE - int(needed_categories[str(annot)])):
            added_train_data.append(img)
            
################################################################################
################################################################################

all_image_files = train_image_files + test_image_files + added_train_data


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


class ImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    

    def __getitem__(self, index):
        image_path = self.paths[index]
        image_l = Image.open(image_path)
        image = image_l.convert('RGB')
        
        if self.transform:
            image_tensor = self.transform(image)
            
        return image_tensor


#dataset = ImageDataset(image_paths, transform)
    
#train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=True)

################################################################################
class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=(224,224), scale=(0.15,0.15)),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.2,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          #transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])


################################################################################


# Load dataset subset
#dataset_train = ImageDataset(train_image_files, 
#                       transform=ContrastiveTransformations(contrast_transforms, n_views=2))

dataset_train = ImageDataset(all_image_files, 
                       transform=ContrastiveTransformations(contrast_transforms, n_views=2))

dataset_val = ImageDataset(test_image_files[0:100000], 
                       transform=ContrastiveTransformations(contrast_transforms, n_views=2))


#sub_train_loader = DataLoader(dataset_sub, batch_size = 100, num_workers = 1, shuffle = True)


#dataset_sub = SimpleImageDataset(sub_image_paths)

print(dataset_train[0][0].shape)


class SimCLR(pl.LightningModule):

    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)  # Output of last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        imgs1, imgs2 = batch
        imgs = torch.cat((imgs1,imgs2), dim=0)
        
        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode+'_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())
        
        wandb.log({"val_acc_top5": (sim_argsort < 5).float().mean(), "loss": nll})
        
        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')
    

def train_simclr(batch_size, max_epochs=2, **kwargs):
    log_dir = "./saved_models/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    trainer = pl.Trainer(default_root_dir=CHECKPOINT_PATH +  "/13_SIM_CLR_VAL",
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True,  mode='max', monitor='val_acc_top5'),
                                    LearningRateMonitor('epoch')])
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = CHECKPOINT_PATH +  '/MB_NEW_SimCLR_Eval_13_Validated.ckpt'
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = SimCLR.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, drop_last = True,num_workers = NUM_WORKERS)
        val_loader = DataLoader(dataset_val, batch_size = batch_size, shuffle = True, num_workers = NUM_WORKERS)
        
        pl.seed_everything(42) # To be reproducable
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) 
        print("Best Path:" + str(trainer.checkpoint_callback.best_model_path))
        trainer.save_checkpoint(pretrained_filename)
        
    return model

wandb.init(
    # set the wandb project where this run will be logged
    project="Contrastive-NHM",
    name = "Model-13-224x224-256btch-300ep-02lowbri-15perc",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 5e-4,
    "architecture": "SimCLR-13",
    "dataset": "Bigger-Herbarium-2022",
    "epochs": 300,
    }
)

# RUN THE TRAINING 
simclr_model = train_simclr(batch_size=128,
                            hidden_dim=128,
                            lr=5e-4,
                            temperature=0.07,
                            weight_decay=1e-4,
                            max_epochs=300)


#torch.save(simclr_model,  save_filename)

wandb.finish()