import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch import optim

def train_one_epoch(model, train_loader:DataLoader, optimizer:optim, criterion:nn, clip:int, device: torch.device)->float:

    model.train()
    
    epoch_loss = 0
    
    for batch in train_loader:
        src, trg = batch
        src, trg = src.permute(1,0).to(device), trg.permute(1,0).to(device)

        optimizer.zero_grad()

        output, _ = model(src, trg[:,:-1])

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)

        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(train_loader)