from time import time
from .train_one_epoch import train_one_epoch
from .evaluate_one_epoch import evaluate_one_epoch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from .utils import epoch_time
import torch
import math

def train_val(epochs:int, train_loader:DataLoader, val_loader:DataLoader, model, optimizer:optim, criterion, clip:int, device: torch.device)->list:
    best_valid_loss = float('inf')
    result = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
    }
    for epoch in range(epochs):

        start_time = time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, clip, device)
        valid_loss = evaluate_one_epoch(model, val_loader, criterion, device)

        end_time = time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        result['epoch'].append(epoch+1)
        result['train_loss'].append(train_loss)
        result['val_loss'].append(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut6-model.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    return best_valid_loss, result