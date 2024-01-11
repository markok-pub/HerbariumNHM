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

#from tensorboard.plugins import projector
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

#print(len(train_image_files))

test_image_files = [join(dirpath,f) for (dirpath, dirnames, filenames) in walk(data_path_test) for f in filenames] 

#print(len(test_image_files))

train_image_files = sorted(train_image_files)
test_image_files = sorted(test_image_files)


import json
 
# Opening JSON file
f = open(ground_truths)
 
# returns JSON object as
# a dictionary
ground_truth_data = json.load(f)

#print(ground_truth_data.keys())
#print(len(ground_truth_data["annotations"]))
#print(ground_truth_data["annotations"][0])
#print(train_image_files[0])

#print(len(ground_truth_data["categories"]))
#print(len(ground_truth_data["genera"]))

#for i, img in enumerate(train_image_files):
#    if "00000__001" in img:
#        print(i)
#        print(img)
        
gt_annot = ground_truth_data["annotations"]
# Iterating through the json
# list 
#Closing file

f.close()

train_data = []
test_data = []

for i, img in enumerate(train_image_files):
    #if(i % 100000 == 0):
        #print(i)
        #print(img)
        #print(gt_annot[i]['image_id'])
    train_data.append((img, gt_annot[i]['category_id']))

#print(len(train_image_files))
#print(len(train_data))

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

#print(len(list(sorted_count.items())))
biggest_categories = list(sorted_count.items())[15001:]
#print(list(sorted_count.items())[12333:])
#print(len(biggest_categories))
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
            reduced_train_data.append((img, dict_keys_back[str(annot)]))

#print(len(reduced_train_data))

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
    
    

class SimCLR(pl.LightningModule):

    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=700):
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
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

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

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')
        

        
model_7 = "./saved_models/contrastive_models/7_SIM_CLR_VAL/lightning_logs/version_1350522/checkpoints/epoch=86-step=1141527.ckpt"        
model_8 = "./saved_models/contrastive_models/8_SIM_CLR_VAL/lightning_logs/version_1358102/checkpoints/epoch=69-step=918470.ckpt"
model_10 = "./saved_models/contrastive_models/10_SIM_CLR_VAL/lightning_logs/version_1436185/checkpoints/epoch=273-step=1797440.ckpt"
model_11 = "./saved_models/contrastive_models/11_SIM_CLR_VAL/lightning_logs/version_1443323/checkpoints/epoch=197-step=1624392.ckpt" 
model_12 = "./saved_models/contrastive_models/12_SIM_CLR_VAL/lightning_logs/version_1472359/checkpoints/epoch=71-step=590688.ckpt"
model_13 = "./saved_models/contrastive_models/13_SIM_CLR_VAL/lightning_logs/version_1527911/checkpoints/epoch=193-step=1591576.ckpt"
model_14 = "./saved_models/contrastive_models/14_SIM_CLR_VAL/lightning_logs/version_1565454/checkpoints/epoch=242-step=2367063.ckpt"
model_15 = "./saved_models/contrastive_models/15_SIM_CLR_VAL/lightning_logs/version_1607025/checkpoints/epoch=202-step=1977423.ckpt"
pretrained_filename = model_11
    
simclr_model = SimCLR.load_from_checkpoint(pretrained_filename)
#simclr_model = torch.load(pretrained_filename)

print(type(simclr_model))
#train_feats_simclr = prepare_data_features(simclr_model, dataset_full_nocut)


class LogisticRegression(pl.LightningModule):

    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=1000):
        super().__init__()
        # Mapping from representation h to classes
        self.save_hyperparameters()
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs*0.6),
                                                                  int(self.hparams.max_epochs*0.8)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode='train'):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + '_loss', loss)
        self.log(mode + '_acc', acc)
        
        #wandb.log({"acc": acc, "loss": loss})
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')
        
        
img_transforms = transforms.Compose([transforms.Resize((1000, 666)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])
#ImageDataset(train_image_files, 
#                       transform=img_transforms)

cpy = reduced_train_data

import random

random.shuffle(cpy)

tr_files = cpy[0:int(0.9*len(cpy))] #train_data[0:543991]
ts_files = cpy[int(0.9*len(cpy)):] #train_data[543991:]

print(len(tr_files))

train_img_data = ImageDataset(tr_files, transform = img_transforms)

test_img_data = ImageDataset(ts_files, transform = img_transforms)


print("Number of training examples:", len(train_img_data))
print("Number of test examples:", len(test_img_data))

print(type(train_img_data[0]))


@torch.no_grad()
def prepare_data_features(model, dataset):
    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        #print(len(batch_imgs))
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return data.TensorDataset(feats, labels)


train_feats_simclr = prepare_data_features(simclr_model, train_img_data)
test_feats_simclr = prepare_data_features(simclr_model, test_img_data)


def train_logreg(batch_size, train_feats_data, test_feats_data, model_suffix, max_epochs=700, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=False,
                         check_val_every_n_epoch=10, 
                            )
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(train_feats_data, batch_size=batch_size, shuffle=True,
                                   drop_last=False, pin_memory=True, num_workers=0)
    test_loader = data.DataLoader(test_feats_data, batch_size=batch_size, shuffle=False,
                                  drop_last=False,  pin_memory=True, num_workers=0)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = LogisticRegression.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LogisticRegression(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        print("Best Path:" + str(trainer.checkpoint_callback.best_model_path))
        
    # Test best model on train and validation set
    train_result = trainer.test(model, train_loader, verbose=True)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}
    
    print(len(test_result))
    print(test_result[0].keys())
    
    return model, result

#wandb.init(
    # set the wandb project where this run will be logged
#    project="NHM-LinReg-Classifier",
#    name="Model11-NEW-777",
    # track hyperparameters and run metadata
#    config={
#    "learning_rate": 1e-3,
#    "architecture": "Model_11_224_lowbri_300-5",
#    "dataset": "Subset-200",
#    "epochs": 1000,
#    }
#)

_, set_results = train_logreg(batch_size=64,
                                        train_feats_data=train_feats_simclr,
                                        test_feats_data=test_feats_simclr,
                                        model_suffix="new_16_3",
                                        feature_dim=train_feats_simclr.tensors[0].shape[1],
                                        num_classes=500,
                                        lr=1e-3,
                                        weight_decay=1e-3)


#save_filename = CHECKPOINT_PATH +  '/Logistic_Regression/logreg_model05_2.ckpt'
#logreg_model_best = deepcopy(_.state_dict())
#torch.save(logreg_model_best.state_dict(), save_filename)

dataset_size = 15000
test_scores = set_results["test"]
train_scores = set_results["train"]

print("Model_11_btch_64_700_500_categories")
print("700 epochs")
print("Train results: " + str(train_scores))
print("Test set results: "+ str(test_scores))

#wandb.finish()