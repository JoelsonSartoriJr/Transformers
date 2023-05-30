from time import time
from train_one_epoch import train_one_epoch
from evaluate_one_epoch import evaluate_one_epoch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from utils import epoch_time
import torch
import math

def train_val(epochs:int, train_loader:DataLoader, val_loader:DataLoader, model:nn.Model, optimizer:optim, criterion:nn, clip:int)->list:
    best_valid_loss = float('inf')

    for epoch in range(epochs):

        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, clip)
        valid_loss = evaluate_one_epoch(model, val_loader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut6-model.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    return best_valid_loss, train_loss, valid_loss