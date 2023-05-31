import re
import random
from collections import Counter
from torchtext.vocab import vocab
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence
import torch

def load_data(path_file:str)->list:
    with open(path_file, encoding='utf-8') as f:
        data = f.read()
        data = re.sub("[^a-zA-Z0-9áàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ\s]","",data) 
        data = re.sub(r"[\u202f]|[\xa0]"," ",data) 
        data = re.sub("([,\.:;!?\"])"," \\1",data) 
        data = re.sub("([¡¿\"])","\\1 ",data).lower()

    data = data.split("\n")
    random.shuffle(data)
    
    data_list = []
    for line in data:
        parts = line.split("\t")
        if len(parts) == 4:
            new_src = [t for t in f'{parts[1]} <eos>'.split(' ') if t]
            new_tgt = [t for t in f'<bos> {parts[3]} <eos>'.split(' ') if t]
            length_src = len(new_src)
            data_list.append((new_src, length_src, new_tgt))
    
    return data_list

def split_data(data:list)->list:
    n = len(data)
    split1, split2 = int(0.7*n), int(0.9*n)
    train_list = data[:split1]
    val_list = data[split1:split2]
    test_list = data[split2:]

    return train_list, val_list, test_list

def create_vocab(data:list)->list:

    counter_src, counter_tgt = Counter(), Counter()
    for i in range(len(data)):
        counter_src.update(data[i][0])
        counter_tgt.update(data[i][-1])

    vocab_src = vocab(counter_src, min_freq = 2,specials=('<unk>', '<eos>', '<bos>', '<pad>'))
    vocab_src.set_default_index(vocab_src['<unk>'])

    vocab_tgt = vocab(counter_tgt, min_freq = 2,specials=('<unk>', '<eos>', '<bos>', '<pad>'))
    vocab_tgt.set_default_index(vocab_tgt['<unk>'])
    
    return vocab_src, vocab_tgt

class BucketSampler(Sampler):
    def __init__(self, batch_size:int, train_list:list)->None:
        self.length = len(train_list)
        self.train_list = train_list
        self.batch_size = batch_size
        indices = [(i, s[1]) for i, s in enumerate(self.train_list)]
        random.shuffle(indices)
        pooled_indices = []
        for i in range(0, len(indices), batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + batch_size * 100],
                                            key=lambda x: x[1], reverse=True))

        self.pooled_indices = pooled_indices

    def __iter__(self)->list:
        for i in range(0, len(self.pooled_indices), self.batch_size):
            yield [idx for idx, _ in self.pooled_indices[i:i + self.batch_size]]

    def __len__(self)->int:
        return (self.length + self.batch_size - 1) // self.batch_size
    
def collate_batch(batch:list, vocab_src:list, vocab_tgt:list, SRC_PAD_IDX:int, TGT_PAD_IDX:int)->tuple:
    text_src, text_tgt = [], []
    for (src, length, tgt) in batch:
        processed_src = torch.tensor([vocab_src[token] for token in src])
        processed_tgt = torch.tensor([vocab_tgt[token] for token in tgt])
        text_src.append(processed_src)
        text_tgt.append(processed_tgt)

    result = (pad_sequence(text_src, padding_value=SRC_PAD_IDX),
                pad_sequence(text_tgt, padding_value=TGT_PAD_IDX))
    return result