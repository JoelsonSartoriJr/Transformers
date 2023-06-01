import random
import numpy as np
import torch
import torch.nn as nn
from time import time
from torchtext.data.metrics import bleu_score
from translate import translate_sentence


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def epoch_time(start_time:time, end_time:time)->int:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model)->int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def calculate_bleu(data:list, src_vocab:list, trg_vocab:list, model, device:torch.device , max_len:int = 50)->float:
    
    trgs = []
    pred_trgs = []
    
    for datum in list(data):
        src, length, trg = datum        
        pred_trg, _ = translate_sentence(src, src_vocab, trg_vocab, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        trg = trg[1:-1]
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)