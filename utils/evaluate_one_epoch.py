import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate_one_epoch(model, val_loader:DataLoader, criterion:nn, device:torch.device)->float:

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(val_loader):
            src, trg = batch
            src, trg = src.permute(1,0).to(device), trg.permute(1,0).to(device)
            output, _ = model(src, trg[:,:-1])

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(val_loader)