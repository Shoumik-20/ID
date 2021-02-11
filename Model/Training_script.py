import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from dataloader import SubsetSC
from data_padding import collate_fn

torchaudio.set_audio_backend("soundfile")

train_set = SubsetSC("testing") #considered the testing set because it was smaller
print(train_set[0])

batch_size = 10
num_workers = 0
pin_memory = False

dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

def train_model(num_epoch = 2):
    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch, num_epoch - 1))
        print("-" * 10)

        for i in dataloader:
            print("This is a batch", i)

train_model(num_epoch = 1)
