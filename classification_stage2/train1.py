#!/usr/bin/env python
# coding: utf-8

# In[1]:


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default
    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    try:
        torch.cuda.set_device(local_rank)
    except:
        print("error at", local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from glob import glob
import copy
import time
import math
import command
import random
import sys
import h5py

#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = 12, 8

from skimage import img_as_ubyte
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import *
from sklearn.metrics import *

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
#import segmentation_models_pytorch_3d as smp
import timm
from torchtoolbox.tools import mixup_data, mixup_criterion
import torchvision as tv
from torch_ema import ExponentialMovingAverage

from transformers import get_cosine_schedule_with_warmup

import torch.distributed as dist

import logging
logging.getLogger('timm').setLevel(logging.WARNING)

#import redis
#import pickle
#redis_cache = redis.StrictRedis(host='localhost', port=6379, db=0)
#redis_cache.flushdb()


# In[ ]:





# In[2]:


class CFG:
    DDP = 1
    DDP_INIT_DONE = 0
    N_GPUS = 2
    FOLD = 0
    FULLDATA = 0
    
    model_name = -1
    V = -1
    
    OUTPUT_FOLDER = f"./data/classification_model/{model_name}_v{V}"
    
    seed = 3407
    
    device = torch.device('cuda')
    
    n_folds = 4
    
    N = 15
    image_size = [384, 384]
    
    train_batch_size = 4
    valid_batch_size = 4
    acc_steps = 1
    
    lr = 1e-4
    wd = 1e-5
    ema_decay_per_epoch = 0.3
    freeze_epochs = 0
    n_epochs = 10
    n_warmup_steps = 0
    upscale_steps = 1.3
    validate_every = 1
    
    epoch = 0
    global_step = 0
    literal_step = 0
    
    autocast = True
    
    workers = 4

if CFG.FULLDATA:
    CFG.seed = CFG.FOLD
    
OUTPUT_FOLDER = CFG.OUTPUT_FOLDER
        
CFG.cache_dir = CFG.OUTPUT_FOLDER + f'/cache/'
os.makedirs(CFG.cache_dir, exist_ok=1)

seed_everything(CFG.seed)


# In[ ]:





# In[3]:


data_master = pd.read_csv('./data/train.csv')
data_master


# In[4]:


train_bbox = pd.read_csv('./data/train_bounding_boxes.csv')
patient_to_slices = {gri: grd.slice_number.values for gri, grd in train_bbox.groupby('StudyInstanceUID')}
train_bbox


# In[5]:


data = pd.read_csv('./data/train_data_XYZcrops.csv')
data


# In[ ]:





# In[6]:


def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0]
    starts = changes[::2]
    lengths = changes[1::2] - starts
    return list(zip(starts, lengths))


def rle_decode(rle, shape):
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    if not len(rle): return mask.reshape(shape)
    
    for start, length in rle:
        mask[start:start + length] = 1
    return mask.reshape(shape)

class SpineDataset(Dataset):
    def __init__(self, data, transforms=None, is_training=False):
        self.data = data
        self.transforms = transforms
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        row = self.data.iloc[i]
        folder = row.folder
        idxs = np.array(eval(row.idxs))
        
        #if 1:
        try:
            volume = np.load(f"./data/train_data_XYZcrops/{folder}/C{row.C-1}_image.npy")
            volume_mask = np.load(f"./data/train_data_XYZcrops/{folder}/C{row.C-1}_mask.npy")
            
            #z = np.linspace(0, volume.shape[0]-1, CFG.N*3).astype(int)
            #volume = volume[z]
            
            z = np.linspace(1, volume.shape[0]-2, CFG.N).astype(int)
            idxs = idxs[z]
            volume = np.concatenate([volume[[x-1, x, x+1]] for x in z])
            
            volume_mask = np.concatenate([volume_mask[[x]] for x in z])
            volume_mask = torch.as_tensor(volume_mask.transpose(0,3,1,2)).float()
            
            volume = volume.reshape(CFG.N, 3, volume.shape[1], volume.shape[2])
            volume = torch.as_tensor(volume).float() / 255
            
            volume = nn.functional.interpolate(volume, (CFG.image_size[0], CFG.image_size[1]))
            volume_mask = nn.functional.interpolate(volume_mask, (256, 256))
            
            label = np.zeros((CFG.N, 2), dtype=np.float32)
            
            #try: label[:, 0] = np.array(eval(row[f"sz"]))[z][::3]
            #except: pass
            
            try: label[:, 0] = np.array(eval(row[f"sz"]))[z]
            except: pass
            
            gt = data_master[data_master.StudyInstanceUID==folder][f"C{row.C}"].values.astype(np.float32)
            
            '''
            if folder in patient_to_slices:
                pos_idxs = patient_to_slices[folder]
                for i, idx in enumerate(idxs):
                    if np.abs(pos_idxs-idx).min() <= 1:
                        label[i, 1] = 1.
            else:
            #'''
            label[:, 1] = gt
            
            
            #label = label * gt
            
            if self.is_training and np.random.random() < 0.5:
                volume = torch.flip(volume, dims=[-1])
                volume_mask = torch.flip(volume_mask, dims=[-1])
                
            if self.is_training and np.random.random() < 0.25:
                volume = torch.flip(volume, dims=[-2])
                volume_mask = torch.flip(volume_mask, dims=[-1])
            
            #volume_mask[:, 0] = torch.clip(volume_mask[:, 0] + volume_mask[:, 1], 0, 1)
            #volume_mask[:, 1] = volume_mask[:, 0]
        #'''    
        except:
            volume = torch.zeros((CFG.N, 3, CFG.image_size[0], CFG.image_size[1])).float()
            volume_mask = torch.zeros((CFG.N, 2, 256, 256)).float()
            label = np.zeros((CFG.N, 2), dtype=np.float32)
        #'''
           
        return {
            'images': volume,
            'labels': label,
            'masks': volume_mask,
            'ids': f"{folder}_{row.C}"
        }


# In[ ]:





# In[7]:


folds = [*GroupKFold(n_splits=CFG.n_folds).split(data, groups=data.folder)]

def get_loaders(ret_data=False, n_workers=CFG.workers):
    train_data = data.iloc[folds[CFG.FOLD][0]]
    valid_data = data.iloc[folds[CFG.FOLD][1]]
    
    #train_data = train_data[:int(len(train_data)*0.2)]
    
    train_transforms = A.Compose([
        A.Resize(*CFG.image_size),
        #A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ])

    valid_transforms = A.Compose([
        A.Resize(*CFG.image_size),
        ToTensorV2()
    ])
    
    train_dataset = SpineDataset(train_data, train_transforms, 1)
    valid_dataset = SpineDataset(valid_data, valid_transforms, 0)
    
    if CFG.DDP and CFG.DDP_INIT_DONE:
        #train_sampler = torch.utils.data.WeightedRandomSampler(weights_list, len(weights_list)//CFG.N_GPUS)
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset, shuffle=True, drop_last=True)
        train_sampler.set_epoch(CFG.epoch) #needed for shuffling?
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, sampler=train_sampler, num_workers=CFG.workers, pin_memory=False, drop_last=True)
        
        valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset=valid_dataset, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, sampler=valid_sampler, shuffle=False, num_workers=CFG.workers, pin_memory=False)
    else:
        #train_sampler = torch.utils.data.WeightedRandomSampler(weights_list, len(weights_list))
        #train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, sampler=train_sampler, num_workers=CFG.workers, pin_memory=False)
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=n_workers, pin_memory=False)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=n_workers, pin_memory=False)
    
    #RUINS VALID-LOADER DDP
    #valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=CFG.workers, pin_memory=False)
    
    CFG.steps_per_epoch = math.ceil(len(train_loader) / CFG.acc_steps)
    
    if ret_data:
        return train_loader, valid_loader, train_data, valid_data
    return train_loader, valid_loader
    
#train_loader, valid_loader = get_loaders(ret_data=False, n_workers=0)
train_loader, valid_loader, train_data, valid_data = get_loaders(ret_data=True, n_workers=0)

#seed_everything(42)

for d in valid_loader: break
    
#_, axs = plt.subplots(2, 4, figsize=(24, 12))
_, axs = plt.subplots(1, 4, figsize=(30, 15))
axs = axs.flatten()
for img, ax in zip(range(4), axs):
    try:
        ax.imshow(d['images'][img][CFG.N//2].numpy().transpose(1,2,0), cmap='gray')
    except: pass
    
_, axs = plt.subplots(1, 4, figsize=(30, 15))
axs = axs.flatten()
for img, ax in zip(range(4), axs):
    try:
        ax.imshow(d['masks'][img][CFG.N//2][0].numpy(), cmap='gray')
    except: pass


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


CFG.n_classes = 2

class SMPModel(nn.Module):
    def __init__(self, pretrained=True, drop=0.):
        super(SMPModel, self).__init__()
        
        segmentor = smp.Unet(encoder_name=f"tu-{CFG.model_name}", encoder_weights='imagenet', in_channels=3, classes=CFG.n_classes, aux_params={'classes': CFG.n_classes})
        #self.segmentor = segmentor
        
        self.encoder = segmentor.encoder
        self.decoder = segmentor.decoder
        self.segmentation_head = segmentor.segmentation_head
        
        self.bridge = nn.Conv2d(256, 1280, kernel_size=(3, 3))
        
        feats = 1280 #self.encoder.num_features
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        lstm_embed = feats
        
        self.lstm = nn.LSTM(lstm_embed, lstm_embed//2, num_layers=2, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Linear(lstm_embed, CFG.n_classes)
        
    def forward(self, inp):
        inp = torch.nan_to_num(inp, 0, 0, 0)

        bs, n_slice_per_c, in_chans, image_size, _ = inp.shape

        inp = inp.view(bs * n_slice_per_c, in_chans, image_size, image_size)

        encoded_features = self.encoder(inp)
        decoded_features = self.decoder(*encoded_features)
        masks = self.segmentation_head(decoded_features)
        
        features = encoded_features[-1]
        features = self.bridge(features)
        
        features = self.avgpool(features)
        features = features.view(bs, n_slice_per_c, -1)

        features, _ = self.lstm(features)
        features = features.contiguous().view(bs * n_slice_per_c, -1)

        logits = self.head(features)
        logits = logits.view(bs, n_slice_per_c, -1).contiguous()
        
        masks = nn.functional.interpolate(masks, (256, 256))
        
        logits = torch.nan_to_num(logits, 0, 0, 0)
        masks = torch.nan_to_num(masks, 0, 0, 0)
        
        return logits, masks
    
class TimmModel(nn.Module):
    def __init__(self, pretrained=True):
        super(TimmModel, self).__init__()
        
        drop = 0.
        self.encoder = timm.create_model(CFG.model_name, pretrained=pretrained, in_chans=3, global_pool='', num_classes=0, drop_rate=drop, drop_path_rate=drop)
        feats = self.encoder.num_features
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        lstm_embed = feats
        
        self.lstm = nn.LSTM(lstm_embed, lstm_embed//2, num_layers=2, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Linear(lstm_embed, CFG.n_classes)
    
    def forward(self, inp):
        inp = torch.nan_to_num(inp, 0, 0, 0)
        
        bs, n_slice_per_c, in_chans, image_size, _ = inp.shape
        
        inp = inp.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        
        features = self.encoder(inp)
        
        features = self.avgpool(features)
        features = features.view(bs, n_slice_per_c, -1)
        
        features, _ = self.lstm(features)
        features = features.contiguous().view(bs * n_slice_per_c, -1)
        
        logits = self.head(features)
        logits = logits.view(bs, n_slice_per_c, -1).contiguous()
        
        logits = torch.nan_to_num(logits, 0, 0, 0)
        
        return logits, None
    
Model = SMPModel


# In[17]:


if CFG.model_name==-1: CFG.model_name = 'tf_efficientnetv2_s.in21k_ft_in1k'

#CFG.model_name = 'tf_efficientnetv2_s.in21k_ft_in1k'
#CFG.model_name = 'maxvit_rmlp_tiny_rw_256.sw_in1k'
#CFG.model_name = 'coat_lite_medium_384.in1k'
#CFG.model_name = 'maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k'
#CFG.model_name = 'tf_efficientnet_b3.ns_jft_in1k'

#'''
model = Model()

model.eval()

#t = time.time()
with torch.no_grad():
    outs = model(d['images'][:1])
#print(time.time() - t)
_ = [print(o.shape) for o in outs if o!=None]
#'''


# In[ ]:





# In[18]:


device = torch.device('cpu')
# change it to nn.BCELoss(reduction='none') if you have sigmoid activation in last layer
loss_fn = nn.BCELoss(reduction="none") 

competition_weights = {
    '-' : torch.tensor([1, 1, 1, 1, 1, 1, 1, 7], dtype=torch.float, device=device),
    '+' : torch.tensor([2, 2, 2, 2, 2, 2, 2, 14], dtype=torch.float, device=device),
}

def competiton_loss(y_hat, y):
    loss = loss_fn(y_hat, y)
    weights = y * competition_weights['+'] + (1 - y) * competition_weights['-']
    loss = (loss * weights).sum(axis=1).mean()

    return loss / weights.sum()

def competiton_loss_row_norm(y_hat, y):
    y_hat, y = torch.as_tensor(y_hat.astype(np.float32)), torch.as_tensor(y.astype(np.float32))
    loss = loss_fn(y_hat, y)
    weights = y * competition_weights['+'] + (1 - y) * competition_weights['-']
    loss = (loss * weights).sum(axis=1)
    w_sum = weights.sum(axis=1)
    loss = torch.div(loss, w_sum)
    return loss.mean()
    
def calculate_competition_metric(OUTPUTS, TARGETS, IDS):
    groups = np.array([x.split('_')[0] for x in IDS])
    unqs = np.unique(groups)

    gt_df = {col: [] for col in ['StudyInstanceUID'] + [f"C{c}" for c in range(1,8)]}
    pred_df = {col: [] for col in ['StudyInstanceUID'] + [f"C{c}" for c in range(1,8)]}

    predictions, labels = [], []
    for group in unqs:

        #prediction = np.concatenate(OUTPUTS3[groups==group])[:, 1].max(0)
        #label = data_master[data_master.StudyInstanceUID==group].values[0][-7:].astype(np.float32)
        prediction = OUTPUTS[groups==group].max(1)[:, 1]
        label = TARGETS[groups==group].max(1)[:, 1]#np.concatenate(TARGETS3[groups==group])[:, 1].max(0)

        predictions.append(prediction)
        labels.append(label)

        gt_df['StudyInstanceUID'].append(group)
        for C in range(1,8):
            try:
                gt_df[f'C{C}'].append(label[C-1])
            except:
                gt_df[f'C{C}'].append(0.)
            
        pred_df['StudyInstanceUID'].append(group)
        for C in range(1,8):
            try:
                pred_df[f'C{C}'].append(prediction[C-1])
            except:
                pred_df[f'C{C}'].append(prediction.mean())
            
        #break

    pred_df, gt_df = pd.DataFrame(pred_df), pd.DataFrame(gt_df)
    pred_df['patient_overall'] = pred_df.values[:, 1:].max(1)
    gt_df['patient_overall'] = gt_df.values[:, 1:].max(1)
    
    preds = pred_df.values[:, 1:]
    labels = gt_df.values[:, 1:]

    preds[:, :7] = np.clip(preds[:, :7], 0, 0.99)
    preds[:, -1] = np.clip(preds[:, -1], 0, 0.99)

    return competiton_loss_row_norm(preds, labels)
    
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([2.]).cuda())
        self.mse = nn.MSELoss()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        
    def forward(self, outputs=None, targets=None, outputs_masks=None, targets_masks=None):
        loss = 0.
        
        #print(outputs.shape, targets.shape, outputs_masks.shape, targets_masks.shape)
        
        if outputs_masks!=None:
            loss1 = self.dice(outputs_masks, targets_masks.flatten(0,1))
            loss = loss + loss1
        if outputs!=None:
            loss2 = self.mse(outputs[:, :, :1], targets[:, :, :1])
            loss3 = self.bce(outputs[:, :, 1:], targets[:, :, 1:])
            #loss4 = self.bce(outputs[:, :, 1:].max(1).values, targets[:, :, 1:].max(1).values)
            
            loss = loss + loss2 + loss3# + loss4
        
        return loss

def plot_lr():
    m = nn.Linear(2, 1)
    optimizer = optim.AdamW(m.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=CFG.steps_per_epoch * CFG.n_epochs * CFG.upscale_steps, num_warmup_steps=CFG.n_warmup_steps)
    
    lrs = []
    for s in range(int(CFG.n_epochs*CFG.steps_per_epoch*CFG.upscale_steps)):
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        lrs.append(lr)
        if s==CFG.n_epochs*CFG.steps_per_epoch:
            break
    return lrs
    
def define_criterion_optimizer_scheduler_scaler(model):
    criterion = CustomLoss().cuda()
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    
    ema_decay_per_iter = CFG.ema_decay_per_epoch ** (1 / CFG.steps_per_epoch)
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay_per_iter)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=CFG.steps_per_epoch * CFG.n_epochs * CFG.upscale_steps, num_warmup_steps=CFG.n_warmup_steps)
    
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.autocast)
    
    return criterion, optimizer, scheduler, scaler, ema


# In[ ]:





# In[19]:


def train_one_epoch(model, loader, running_dist=True):
    model.train()
    running_loss = 0.0

    if is_main_process(): bar = tqdm(loader, bar_format='{n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}')
    else: bar = loader
    
    for step, data in enumerate(bar):
        step += 1
        
        images = data['images'].cuda()
        targets = data['labels'].cuda()
        targets_masks = data['masks'].cuda()
        
        with torch.cuda.amp.autocast(enabled=CFG.autocast, dtype=torch.float16):
            logits, logits_masks = model(images)
            
            loss = criterion(logits, targets, logits_masks, targets_masks)
        
        running_loss += (loss - running_loss) * (1 / step)
        
        loss = loss / CFG.acc_steps
        scaler.scale(loss).backward()
        
        if step % CFG.acc_steps == 0 or step == len(bar):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            ema.update()
            CFG.global_step += 1
        
        CFG.literal_step += 1
        
        lr = "{:2e}".format(optimizer.param_groups[0]['lr'])
        
        if is_main_process():
            bar.set_postfix(loss=running_loss.item(), lr=float(lr), step=CFG.global_step)
        
        if running_dist:
            dist.barrier()
        
        #if step==10: break
    
    if is_main_process():
        if running_dist:
            torch.save(model.module.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth")
            
        else:
            torch.save(model.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth")
        
        
def valid_one_epoch(path, loader, running_dist=True, debug=False, do_ema=True):
    #model = Model(pretrained=False)
    #st = torch.load(path, map_location=f"cpu")
    model.eval()
    model.cuda()
    #model.load_state_dict(st, strict=False)
    
    if is_main_process(): bar = tqdm(loader, bar_format='{n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}')
    else: bar = loader
    
    running_loss = 0.
    
    OUTPUTS, TARGETS, IDS = [], [], []
    
    for step, data in enumerate(bar):
        
        with torch.no_grad():
            images = data['images'].cuda()
            targets = data['labels'].cuda()
            targets_masks = None#data['masks'].cuda()
            ids = data['ids']
            
            with torch.cuda.amp.autocast(enabled=CFG.autocast):
                if do_ema:
                    with ema.average_parameters():
                        logits, logits_mask = model(images)
                else:
                    logits, logits_mask = model(images)
                
            outputs = logits.float().sigmoid().detach().cpu().numpy()
            targets = targets.float().detach().cpu().numpy()
            #outputs = logits_mask.float().sigmoid().detach().cpu().numpy()
            #targets = targets_masks.float().detach().cpu().numpy()
            
            #'''
            if running_dist:
                dist.barrier()
                
                np.save(f'{CFG.cache_dir}/preds_{get_rank()}.npy', outputs)
                np.save(f'{CFG.cache_dir}/targets_{get_rank()}.npy', targets)
                np.save(f'{CFG.cache_dir}/ids_{get_rank()}.npy', ids)
                
                dist.barrier()
                
                if is_main_process():
                    outputs = np.concatenate([np.load(f"{CFG.cache_dir}/preds_{_}.npy") for _ in range(CFG.N_GPUS)])
                    targets = np.concatenate([np.load(f"{CFG.cache_dir}/targets_{_}.npy") for _ in range(CFG.N_GPUS)])
                    ids = np.concatenate([np.load(f"{CFG.cache_dir}/ids_{_}.npy") for _ in range(CFG.N_GPUS)])
                    
                dist.barrier()
            else:    
                pass
            
            OUTPUTS.extend(outputs)
            TARGETS.extend(targets)
            IDS.extend(ids)
            #'''
            
            #outputs = np.concatenate(outputs.transpose(0, 4, 1, 2, 3), 0)
            #targets = np.concatenate(targets.transpose(0, 4, 1, 2, 3), 0)
            
            #PixelLevelF1Scorer.accumulate(outputs, targets)
            
            #if step==10: break
            
    if running_dist:
        dist.barrier()
    
    if is_main_process():
        OUTPUTS = np.stack(OUTPUTS)#[:, :6]
        TARGETS = np.stack(TARGETS)#[:, :6]
        IDS = np.stack(IDS)
        
        #return OUTPUTS, TARGETS, IDS
        
        #score = PixelLevelF1Scorer.calculate_score()
        #score, classwise = DiceScorer.calculate_score()
        
        score = roc_auc_score(TARGETS.max(1)[:, 1], OUTPUTS.max(1)[:, 1])
        
        loss = calculate_competition_metric(OUTPUTS, TARGETS, IDS)
        
        print(f"EPOCH {CFG.epoch+1} | AUC {score} LOSS {loss}")
        #print(classwise)
        
        if debug:
            return loss, OUTPUTS, TARGETS, IDS
    
        return loss
    
    if debug:
        return [], [], [], []
    
def run(model, get_loaders):
    if is_main_process():
        epochs = []
        scores = []
    
    best_score = float('inf')
    for epoch in range(CFG.n_epochs):
        CFG.epoch = epoch
        
        train_loader, valid_loader = get_loaders()
        
        '''
        if CFG.epoch<CFG.freeze_epochs:
            for param in model.module.parameters():
                param.requires_grad = False
        else:
            for param in model.module.parameters():
                param.requires_grad = True
        #'''
        
        train_one_epoch(model, train_loader, running_dist=CFG.DDP_INIT_DONE)
        
        if CFG.DDP_INIT_DONE:
            dist.barrier()
        
        if (CFG.epoch+1)%CFG.validate_every==0 or epoch==0:
            if is_main_process():
                with ema.average_parameters():
                    torch.save(model.module.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}_EMA.pth")
                    
            if CFG.DDP_INIT_DONE:
                dist.barrier()
                
            #score, OUTPUTS, TARGETS, IDS  = valid_one_epoch(f"{OUTPUT_FOLDER}/{CFG.FOLD}_EMA.pth", valid_loader, debug=True, running_dist=CFG.DDP_INIT_DONE)
            score = valid_one_epoch(f"{OUTPUT_FOLDER}/{CFG.FOLD}_EMA.pth", valid_loader, debug=False, running_dist=CFG.DDP_INIT_DONE)
            
        if CFG.DDP_INIT_DONE:
            dist.barrier()
        
        if is_main_process():
            epochs.append(epoch)
            scores.append(score)
            
            if score <= best_score:
                print("SAVING BEST!")
                if CFG.DDP_INIT_DONE:
                    with ema.average_parameters():
                        torch.save(model.module.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}_best.pth")
                else:
                    with ema.average_parameters():
                        torch.save(model.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}_best.pth")
                    
                best_score = score
                
                #np.save(f"{OUTPUT_FOLDER}/OUTPUTS_{CFG.FOLD}.npy", OUTPUTS)
                #np.save(f"{OUTPUT_FOLDER}/TARGETS_{CFG.FOLD}.npy", TARGETS)
                #np.save(f"{OUTPUT_FOLDER}/IDS_{CFG.FOLD}.npy", IDS)                
            
            try:
                command.run(['rm', '-r', CFG.cache_dir])
                pass
            except:
                pass
            
            os.makedirs(CFG.cache_dir, exist_ok=1)


# In[ ]:





# In[ ]:


CFG.DDP = 1

if __name__ == '__main__' and CFG.DDP:
    
    world_size = init_distributed()
    CFG.DDP_INIT_DONE = 1
    
    #important to setup before defining scheduler to establish the correct number of steps per epoch
    train_loader, valid_loader = get_loaders()
    
    model = Model().cuda()
    
    #st = torch.load(f'./data/classification_stage1/tf_efficientnetv2_s.in21k_ft_in1k_v6/{CFG.FOLD}_best.pth', map_location='cpu')
    #st = torch.load(f'./data/classification_stage1/tf_efficientnet_b3.ns_jft_in1k_v2/0_best.pth', map_location='cpu')
    st = torch.load(f'./data/classification_stage1/tf_efficientnetv2_s.in21k_ft_in1k_v7/0_best.pth', map_location='cpu')
    new_st = {}
    for key in st:
        if 'encoder' in key:
            new_st[key.replace('encoder.', 'model.')] = st[key]
    model.encoder.load_state_dict(new_st, strict=False)
    
    #st = torch.load(f'./data/AAA_CLS/TRY8_CLS/tf_efficientnetv2_s.in21k_ft_in1k_v1/{CFG.FOLD}_best.pth')
    #for key in ['head.weight', 'head.bias']: del st[key]
    #model.load_state_dict(st, strict=False)
    
    #if is_main_process():
        #torch.save(model.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth")
        
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    local_rank = int(os.environ['LOCAL_RANK'])
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    
    #model = torch.compile(model)
    
    criterion, optimizer, scheduler, scaler, ema = define_criterion_optimizer_scheduler_scaler(model)
    
    run(model, get_loaders)
    
else:
    
    CFG.DDP_INIT_DONE = False
    #important to setup before defining scheduler to establish the correct number of steps per epoch
    train_loader, valid_loader = get_loaders()
    
    model = Model().cuda()
    
    #model.load_state_dict(torch.load(f"/mnt/md0/rsna_spine/AAA_CLS/TRY12_CLS/b5_v1/best_f{CFG.FOLD}.pth", map_location='cpu'), strict=False)
    
    criterion, optimizer, scheduler, scaler = define_criterion_optimizer_scheduler_scaler(model)
    
    run(model, get_loaders)
    
import sys
sys.exit(0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1053]:


model = Model()
st = torch.load('./data/classification_model/tf_efficientnetv2_s.in21k_ft_in1k_v10/1_best.pth')
model.load_state_dict(st)


# In[ ]:





# In[1475]:


OUTPUTS, TARGETS, IDS = [], [], []

for F in range(4):
    CFG.FOLD = F
    
    model = Model()
    st = torch.load(f'./data/classification_model/tf_efficientnetv2_s.in21k_ft_in1k_v12/{F}_best.pth')
    model.load_state_dict(st)
    
    train_loader, valid_loader = get_loaders(n_workers=4)
    score, outputs, targets, ids = valid_one_epoch('', valid_loader, running_dist=False, do_ema=False, debug=True)
    
    OUTPUTS.extend(outputs)
    TARGETS.extend(targets)
    IDS.extend(ids)

OUTPUTS, TARGETS, IDS = np.stack(OUTPUTS), np.stack(TARGETS), np.stack(IDS)
OUTPUTS.shape, TARGETS.shape, IDS.shape


# In[ ]:





# In[1476]:


groups = np.array([x.split('_')[0] for x in IDS])
unqs = np.unique(groups)

gt_df = {col: [] for col in ['StudyInstanceUID'] + [f"C{c}" for c in range(1,8)]}
pred_df = {col: [] for col in ['StudyInstanceUID'] + [f"C{c}" for c in range(1,8)]}

predictions, labels = [], []
for group in unqs:
    
    #prediction = np.concatenate(OUTPUTS3[groups==group])[:, 1].max(0)
    #label = data_master[data_master.StudyInstanceUID==group].values[0][-7:].astype(np.float32)
    prediction = OUTPUTS[groups==group].max(1)[:, 1]
    label = TARGETS[groups==group].max(1)[:, 1]#np.concatenate(TARGETS3[groups==group])[:, 1].max(0)
    
    predictions.append(prediction)
    labels.append(label)
    
    gt_df['StudyInstanceUID'].append(group)
    for C in range(1,8):
        try:
            gt_df[f'C{C}'].append(label[C-1])
        except:
            gt_df[f'C{C}'].append(label[0])
    
    pred_df['StudyInstanceUID'].append(group)
    for C in range(1,8):
        try:
            pred_df[f'C{C}'].append(prediction[C-1])
        except:
            pred_df[f'C{C}'].append(prediction.mean())
        
    #break

pred_df, gt_df = pd.DataFrame(pred_df), pd.DataFrame(gt_df)
pred_df['patient_overall'] = pred_df.values[:, 1:].max(1)
gt_df['patient_overall'] = gt_df.values[:, 1:].max(1)


# In[1477]:


pred_df


# In[1478]:


gt_df


# In[1479]:


for C in range(1,8):
    print(C, pred_df[gt_df[f'C{C}']==1][f"C{C}"].mean(), pred_df[gt_df[f'C{C}']==0][f"C{C}"].mean())


# In[ ]:





# In[1480]:


device = torch.device('cpu')
# change it to nn.BCELoss(reduction='none') if you have sigmoid activation in last layer
loss_fn = nn.BCELoss(reduction="none") 

competition_weights = {
    '-' : torch.tensor([1, 1, 1, 1, 1, 1, 1, 7], dtype=torch.float, device=device),
    '+' : torch.tensor([2, 2, 2, 2, 2, 2, 2, 14], dtype=torch.float, device=device),
}

def competiton_loss(y_hat, y):
    loss = loss_fn(y_hat, y)
    weights = y * competition_weights['+'] + (1 - y) * competition_weights['-']
    loss = (loss * weights).sum(axis=1).mean()

    return loss / weights.sum()

def competiton_loss_row_norm(y_hat, y):
    y_hat, y = torch.as_tensor(y_hat.astype(np.float32)), torch.as_tensor(y.astype(np.float32))
    loss = loss_fn(y_hat, y)
    weights = y * competition_weights['+'] + (1 - y) * competition_weights['-']
    loss = (loss * weights).sum(axis=1)
    w_sum = weights.sum(axis=1)
    loss = torch.div(loss, w_sum)
    return loss.mean()


# In[1483]:


preds = pred_df.values[:, 1:].copy()
labels = gt_df.values[:, 1:].copy()

preds[:, :7] = np.clip(preds[:, :7] * 1., 0., 0.99)
preds[:, -1] = np.clip(preds[:, -1] * 1., 0., 0.99)

competiton_loss_row_norm(preds, labels)


# In[ ]:





# In[ ]:





# In[ ]:





# In[1217]:


preds = pred_df.values[:, 1:].copy()
labels = gt_df.values[:, 1:].copy()

preds[:, :7] = np.clip(preds[:, :7] * 1., 0., 0.99)
preds[:, -1] = np.clip(preds[:, -1] * 1., 0., 1.)

competiton_loss_row_norm(preds, labels)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[832]:


np.sum(gt_df.patient_overall==(pred_df.patient_overall>0.2)) / len(gt_df)


# In[ ]:





# In[ ]:





# In[730]:


roc_auc_score(labels, predictions)


# In[ ]:





# In[736]:


predictions[labels==1].mean(), predictions[labels==0].mean()


# In[582]:


predictions[labels==1].mean(), predictions[labels==0].mean()


# In[ ]:





# In[611]:


roc_auc_score(labels, predictions)


# In[574]:


roc_auc_score(labels, predictions)


# In[563]:


roc_auc_score(labels, predictions)


# In[433]:


roc_auc_score(labels, predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[202]:


_ = valid_one_epoch('', valid_loader, running_dist=False, do_ema=False, debug=True)


# In[207]:


score, OUTPUTS, TARGETS, IDS = _
OUTPUTS = np.stack(OUTPUTS)
TARGETS = np.stack(TARGETS)
IDS = np.array(IDS)
OUTPUTS.shape, TARGETS.shape, IDS.shape


# In[ ]:





# In[209]:


data_master


# In[ ]:





# In[385]:


groups = np.array([x.split('_')[0] for x in IDS])
unqs = np.unique(groups)

predictions, labels = [], []
for group in unqs:
    
    prediction = np.concatenate(OUTPUTS[groups==group]).mean(0)
    label = data_master[data_master.StudyInstanceUID==group].values[0][-7:].astype(np.float32)
    #label = np.concatenate(TARGETS[groups==group]).max(0)
    
    predictions.append(prediction)
    labels.append(label)
    
    #break
predictions, labels = np.array(predictions), np.array(labels)


# In[386]:


np.sum(labels>0.5) / (len(labels) * 7)


# In[ ]:





# In[380]:


roc_auc_score(labels>0.5, predictions)


# In[ ]:





# In[314]:


tgs.shape


# In[ ]:





# In[345]:


data_master[data_master.C7==1]


# In[ ]:





# In[358]:


IDS[groups=='1.2.826.0.1.3680043.7824']


# In[ ]:





# In[ ]:





# In[369]:


np.concatenate(TARGETS[groups=='1.2.826.0.1.3680043.7824'], 0)[:, -1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[342]:


np.sum(tgs[:, -1] > 0.) / len(tgs)


# In[330]:


np.sum(tgs > 0.) / (110784 * 7)


# In[340]:


data_master.values[:, -7:].sum() / (len(data_master) * 7)


# In[ ]:





# In[327]:


data_master.C7.sum() / len(data_master)


# In[ ]:





# In[ ]:





# In[295]:


labels


# In[ ]:





# In[ ]:





# In[285]:


roc_auc_score(np.concatenate(TARGETS) > 0.5, np.concatenate(OUTPUTS))


# In[ ]:





# In[251]:


predictions[labels==0].mean()


# In[249]:


predictions[labels==1].mean()


# In[ ]:





# In[245]:


predictions[1]


# In[260]:


tgs = np.concatenate(TARGETS)
outs = np.concatenate(OUTPUTS)
tgs.shape, outs.shape


# In[265]:


np.sum(tgs > 0.1)


# In[272]:


outs[tgs < 0.1].mean(), outs[tgs > 0.1].mean()


# In[267]:


roc_auc_score(np.concatenate(TARGETS)>0.5, np.concatenate(OUTPUTS))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[214]:


IDS[groups==group]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[142]:


_ = valid_one_epoch('', valid_loader, running_dist=False, do_ema=False)


# In[132]:


np.concatenate(outputs.transpose(0, 4, 1, 2, 3), 0).shape


# In[ ]:





# In[127]:


outputs.shape, targets.shape


# In[116]:


plt.imshow(targets[3][0][:, :, 31])


# In[115]:


plt.imshow(outputs[3][0][:, :, 31])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


def test_one_epoch(path, loader, running_dist=True, debug=False, do_ema=True):
    model = Model(pretrained=False)
    st = torch.load(path, map_location=f"cpu")
    model.eval()
    model.cuda()
    model.load_state_dict(st, strict=False)
    
    if is_main_process(): bar = tqdm(loader, bar_format='{n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}')
    else: bar = loader
    
    running_loss = 0.
    
    OUTPUTS, TARGETS, IDS = [], [], []
    
    for step, data in enumerate(bar):
        
        with torch.no_grad():
            images = data['images'].cuda()
            targets = data['labels'].cuda()
            #targets_masks = data['masks'].cuda()
            ids = data['ids']
            
            with torch.cuda.amp.autocast(enabled=CFG.autocast):
                if do_ema:
                    with ema.average_parameters():
                        logits, logits_mask = model(images)
                else:
                    logits, logits_mask = model(images)
                
            return logits, logits_mask
                
            outputs = logits.float().detach().cpu().numpy()
            targets = targets.float().detach().cpu().numpy()
            #outputs = logits_mask.float().sigmoid().detach().cpu()#.numpy()
            #targets = targets_masks.float().detach().cpu()#.numpy()
            
            #'''
            if running_dist:
                dist.barrier()
                
                np.save(f'{CFG.cache_dir}/preds_{get_rank()}.npy', outputs)
                np.save(f'{CFG.cache_dir}/targets_{get_rank()}.npy', targets)
                np.save(f'{CFG.cache_dir}/ids_{get_rank()}.npy', ids)
                
                dist.barrier()
                
                if is_main_process():
                    outputs = np.concatenate([np.load(f"{CFG.cache_dir}/preds_{_}.npy") for _ in range(CFG.N_GPUS)])
                    targets = np.concatenate([np.load(f"{CFG.cache_dir}/targets_{_}.npy") for _ in range(CFG.N_GPUS)])
                    ids = np.concatenate([np.load(f"{CFG.cache_dir}/ids_{_}.npy") for _ in range(CFG.N_GPUS)])
                    
                dist.barrier()
            else:    
                pass
            
            OUTPUTS.extend(outputs)
            TARGETS.extend(targets)
            IDS.extend(ids)
            #'''
            
            #if step==10: break
            
    if running_dist:
        dist.barrier()
    
    if is_main_process():
        OUTPUTS = np.stack(OUTPUTS)#[:, :6]
        TARGETS = np.stack(TARGETS)#[:, :6]
        IDS = np.stack(IDS)
        
        #return OUTPUTS, TARGETS, IDS
        
        score = np.mean(np.abs(OUTPUTS-TARGETS))
        
        print(f"EPOCH {CFG.epoch+1} | MAE {score}")
    
        if debug:
            return score, OUTPUTS, TARGETS, IDS
    
        return score
    
    if debug:
        return [], [], [], []


# In[ ]:





# In[39]:


logits, logits_mask = test_one_epoch('./data/AAA_SEG/TRY1_SEG/tf_efficientnetv2_s.in21k_ft_in1k_v2/3_best.pth', valid_loader, do_ema=False, debug=True, running_dist=False)


# In[ ]:





# In[46]:


x = 5
plt.imshow(d['masks'][x][1]); plt.show()
plt.imshow(logits_mask.sigmoid()[x][1].detach().cpu()); plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


score, OUTPUTS, TARGETS, IDS = test_one_epoch('./data/AAA_SEG/TRY1_SEG/tf_efficientnetv2_s.in21k_ft_in1k_v1/0_best.pth', valid_loader, do_ema=False, debug=True, running_dist=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[100]:


np.abs(np.round(OUTPUTS[:, 0]) - TARGETS[:, 0]).mean()


# In[99]:


np.abs(np.round(OUTPUTS[:, 1]) - TARGETS[:, 1]).mean()


# In[ ]:





# In[104]:


np.abs(TARGETS[:, 0].mean() - TARGETS[:, 0]).mean()


# In[105]:


np.abs(TARGETS[:, 1].mean() - TARGETS[:, 1]).mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[95]:


np.round(OUTPUTS[:, 1], 0)


# In[93]:


TARGETS[:, 1]


# In[ ]:





# In[ ]:





# In[82]:


TARGETS[:, 1].mean()


# In[ ]:





# In[ ]:





# In[20]:


score, OUTPUTS, TARGETS, IDS = test_one_epoch('./data/AAA_CLS/TRY1_CLS/tf_efficientnetv2_s.in21k_ft_in1k_v2/0_best.pth', valid_loader, do_ema=False, debug=True, running_dist=False)


# In[ ]:





# In[26]:


OUTPUTS[TARGETS==0]


# In[25]:


np.sum(TARGETS==0, 0)


# In[37]:


TARGETS[(TARGETS[:, 0]==0) & (TARGETS[:, 2]!=1.)]


# In[35]:


TARGETS[(TARGETS[:, 1]==0) & (TARGETS[:, 2]!=1.)]


# In[ ]:





# In[42]:


np.round(OUTPUTS[(TARGETS[:, 0]==0) & (TARGETS[:, 2]!=1.)], 3)


# In[ ]:





# In[58]:


np.round(OUTPUTS[(TARGETS[:, 1]==0) & (TARGETS[:, 2]!=1.)], 3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


OUTPUTS[TARGETS==1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




