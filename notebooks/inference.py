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
from tensorboard.plugins import projector
import cv2
import pathlib
import os
import datetime
from os import listdir, walk
from os.path import isfile, join
import wandb
import random

#IMPORT TEST FILES

data_path_test = "/gpfs/data/fs71186/kadic/Herbarium_2022/test_images"

test_image_files = [join(dirpath,f) for (dirpath, dirnames, filenames) in walk(data_path_test) for f in filenames] 

print(len(test_image_files))

test_image_files = sorted(test_image_files)

ground_truths = "/gpfs/data/fs71186/kadic/Herbarium_2022/test_metadata.json"

import json
 
# Opening JSON file
f = open(ground_truths)
 
# returns JSON object as
# a dictionary
ground_truth_data = json.load(f)
print(type(ground_truth_data))
f.close()

image_ids = {}

for i, img_path in enumerate(test_image_files):
    image_ids[img_path] = ground_truth_data[i]["image_id"]
    if(i%100000 == 0):
        print(img_path)
        print(ground_truth_data[i]["file_name"])

#SET UP
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

## DATASET CLASS

class TestImageDataset(Dataset):
    def __init__(self, paths,transform):
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        image_path = self.paths[index]
        image_l = Image.open(image_path)
        image = image_l.convert('RGB')
        image_tensor = image
        if self.transform:
            image_tensor = self.transform(image)
        
        return image_tensor
    
## SIMCLR CLASS TO LOAD THE MODEL

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
        
# LOAD SIMCLR MODEL
model_11 = "./saved_models/contrastive_models/11_SIM_CLR_VAL/lightning_logs/version_1443323/checkpoints/epoch=197-step=1624392.ckpt"
model_14 = "./saved_models/contrastive_models/14_SIM_CLR_VAL/lightning_logs/version_1565454/checkpoints/epoch=242-step=2367063.ckpt"
model_15 = "./saved_models/contrastive_models/15_SIM_CLR_VAL/lightning_logs/version_1607025/checkpoints/epoch=202-step=1977423.ckpt"

pretrained_filename = model_15
    
simclr_model = SimCLR.load_from_checkpoint(pretrained_filename)

print(type(simclr_model))

# LOGREG CLASS 

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
        
        wandb.log({"acc": acc, "loss": loss})
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model(batch[0])
    

img_transforms = transforms.Compose([transforms.Resize((500, 333)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])

test_img_data = TestImageDataset(test_image_files, transform = img_transforms)

@torch.no_grad()
def prepare_data_features(model, dataset):
    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    feats = []
    for batch_imgs in tqdm(data_loader):
        #print(len(batch_imgs))
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())

    feats = torch.cat(feats, dim=0)

    return data.TensorDataset(feats)

# PREPARE TEST SET DATA FEATURES
test_feats_simclr = prepare_data_features(simclr_model, test_img_data)

# LOAD UP THE LOGISTIC REGRESSION MODEL
model_14_logreg = "./saved_models/contrastive_models/LogisticRegression/lightning_logs/version_1614704/checkpoints/epoch=609-step=8004420.ckpt"

model_15_logreg = "./saved_models/contrastive_models/LogisticRegression/lightning_logs/version_1617998/checkpoints/epoch=609-step=8004420.ckpt"

model_11_new = "./saved_models/contrastive_models/LogisticRegression/lightning_logs/version_1632116/checkpoints/epoch=639-step=7978240.ckpt"

pretrained_filename = model_15_logreg

if os.path.isfile(pretrained_filename):
    print(f"Found pretrained model at {pretrained_filename}, loading...")
    model = LogisticRegression.load_from_checkpoint(pretrained_filename)
    

# DO THE INFERENCE
test_loader = data.DataLoader(test_feats_simclr, batch_size=64, shuffle=False,
                                   drop_last=False, pin_memory=True, num_workers=64)

trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression_inf"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,)

predictions = trainer.predict(model, test_loader)

predictions = torch.cat(predictions,dim = 0)

true_preds = predictions.argmax(dim=-1)

print(len(true_preds))
print(true_preds[0])
print("finished predicting")
# SAVE THE RESULTS TO A CSV

frame = {"Id" : [], "Predicted": [] }

df = pandas.DataFrame(frame)
        
for i,pred in enumerate(true_preds):
    result_one = pred.item()
    if(result_one == 103):
        result_one = 15501
    if(result_one == 2683):
        result_one = 15502
    if(result_one == 7000):
        result_one = 15503
    if(result_one == 9000):
        result_one = 15504
    row = {"Id" : image_ids[test_image_files[int(i)]], "Predicted": str(result_one) }
    df.loc[len(df)] = row

df.to_csv('./predictions/preds_model_15_2.csv', index = False)
print(len(true_preds))
