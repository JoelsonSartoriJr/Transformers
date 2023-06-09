from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from utils.train_val import train_val
from utils.utils import set_seed, count_parameters, initialize_weights
from utils.data import load_data, split_data, create_vocab, BucketSampler, collate_batch
from utils.graphs import *
from utils.translate import translate_sentence
from torchtext.data.metrics import bleu_score
from transformers import Encoder
from transformers import Decoder
from seq2seq import Seq2Seq
## Set seed
SEED = 1711
set_seed(SEED)

#Function
def collate_batch(batch:list)->tuple:
    text_src, text_tgt = [], []
    for (src, length, tgt) in batch:
        processed_src = torch.tensor([vocab_src[token] for token in src])
        processed_tgt = torch.tensor([vocab_tgt[token] for token in tgt])
        text_src.append(processed_src)
        text_tgt.append(processed_tgt)

    result = (pad_sequence(text_src, padding_value=SRC_PAD_IDX),
                pad_sequence(text_tgt, padding_value=TGT_PAD_IDX))
    return result

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

## Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH_DATA = 'data/por.txt'
batch_size = 64 

data = load_data(PATH_DATA)
train_list, val_list, test_list = split_data(data)
vocab_src, vocab_tgt = create_vocab(train_list)

SRC_PAD_IDX = vocab_src['<pad>']
TGT_PAD_IDX = vocab_tgt['<pad>']

train_bucket = BucketSampler(batch_size, train_list)
train_iter = DataLoader(train_list,
                            batch_sampler=train_bucket,
                            collate_fn=collate_batch)

val_bucket = BucketSampler(batch_size, val_list)
val_iter = DataLoader(val_list,
                        batch_sampler=val_bucket,
                        collate_fn=collate_batch)

test_bucket = BucketSampler(batch_size, test_list)
test_iter = DataLoader(test_list,
                            batch_sampler=test_bucket,
                            collate_fn=collate_batch)

#Model
INPUT_DIM = len(vocab_src.get_itos())
OUTPUT_DIM = len(vocab_tgt.get_itos())
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
LEARNING_RATE = 0.0005
N_EPOCHS = 10
CLIP = 1

enc = Encoder(INPUT_DIM, 
                HID_DIM, 
                ENC_LAYERS, 
                ENC_HEADS, 
                ENC_PF_DIM, 
                ENC_DROPOUT, 
                DEVICE)

dec = Decoder(OUTPUT_DIM, 
                HID_DIM, 
                DEC_LAYERS, 
                DEC_HEADS, 
                DEC_PF_DIM, 
                DEC_DROPOUT, 
                DEVICE)

SRC_PAD_IDX = vocab_src['<pad>']
TGT_PAD_IDX = vocab_tgt['<pad>']

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TGT_PAD_IDX, DEVICE).to(DEVICE)

print(f'The model has {count_parameters(model):,} trainable parameters')

model.apply(initialize_weights)

optimizer = Adam(model.parameters(), lr = LEARNING_RATE)
criterion = CrossEntropyLoss(ignore_index = TGT_PAD_IDX)

best, result = train_val(N_EPOCHS, train_iter, val_iter, model, optimizer, criterion, CLIP, DEVICE)

error_plot(result)

example_idx = 20

src, length, trg  = train_list[example_idx]
print(f'src = {src}')
print(f'trg = {trg}')

translation, attention = translate_sentence(src, vocab_src, vocab_tgt, model, DEVICE)

print(f'predicted trg = {translation}')

display_attention(src, translation, attention)

bleu_score = calculate_bleu(test_list, vocab_src, vocab_tgt, model, DEVICE)

print(f'BLEU score = {bleu_score*100:.2f}')