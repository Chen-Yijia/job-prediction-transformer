#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:44:35 2020

@author: angelica_cyj
"""

import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''-------------------Prepare the data----------------------'''

'''train_set'''

def ToWord(title):
    target_words=[]
    for word in title.split(' '):
        if word != '' and word != ' ':
            target_words.append(word)
        
    return target_words

def PrepareTrainSet():
    with open('Book3.csv','r') as r:
        reader=csv.reader(r)
        index=0
        train_set={}

            
        for row in reader:
            person=[] 
            title_sequence=[]
            #delete the empty element
            for element in row:
                if len(element) == 0:
                    break
                else:
                    person.append(element) #person: id+job_title_sequence
            title_sequence=person[1:]      #title_sequnce: job sequence without id
            max_length=len(title_sequence)-1
            for i in range(1,max_length+1):
                for j in range(0,len(title_sequence)-i):
                    data={}
                    data['input']=[]
                    for k in range(j,i+j):
                        data['input'].append(title_sequence[k])
#                        data['input'].append(str(EOS_token))
                    title=title_sequence[i+j]
                    data['target']=ToWord(title)
#                    data['target'].append(str(EOS_token))
                    train_set[index]=data
                    index += 1
           
    return train_set

train_set = PrepareTrainSet()
lenTrainSet=len(train_set)

'''title bank & word bank'''

class Word():
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addTrainSet(self, train_set):
#        for word in sentence.split(' '):
#            self.addWord(word)
        for i in range(len(train_set)):
            for word in train_set[i]['target']:
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
wordbank=Word()
wordbank.addTrainSet(train_set)

'''title: word embedding'''

from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
import flair

flair.device = device
flair.cache_root='./cache'

# initialize the word embeddings
glove_embedding = WordEmbeddings('glove')

# initialize the document embeddings, mode = mean
doc_embedding = DocumentPoolEmbeddings([glove_embedding])    # embedding size: 100

def wordEmbedding(title_list):
    for i in range(len(title_list)):
        if i == 0:
            try:
                sentence=Sentence(title_list[i])
                doc_embedding.embed(sentence)
                previous_tensor=sentence.embedding.view(1,1,-1)
            except:
                sentence = Sentence('unknown')
                doc_embedding.embed(sentence)
                previous_tensor=sentence.embedding.view(1,1,-1)
        else:
            try:
                sentence=Sentence(title_list[i])
                doc_embedding.embed(sentence)
                title_tensor=sentence.embedding.view(1,1,-1)
            except:
                sentence = Sentence('unknown')
                doc_embedding.embed(sentence)
                title_tensor=sentence.embedding.view(1,1,-1)
                
            previous_tensor = torch.cat((previous_tensor,title_tensor),0)
            
    return previous_tensor




'''Dataset'''

SOS_token = 0
EOS_token = 1

class PositionDataset(Dataset):
    
    def __init__(self,train_set):
        self.train_set=train_set
        
    def __len__(self):
        return len(self.train_set)
    
    def __getitem__(self,index):
        target_indexes=[SOS_token]
        
        input_tensor = wordEmbedding(self.train_set[index]['input'])
        
        for word in self.train_set[index]['target']:
            target_indexes.append(wordbank.word2index[word])
        target_indexes.append(EOS_token)
        target_tensor = torch.tensor(target_indexes, dtype = torch.long, device = device).view(-1,1)
        
        sample={'input':input_tensor, 'target':target_tensor}
        # input tensor: tensor(index1,index2...)
      
        return sample

positionsDataset = PositionDataset(train_set)
   
    
    
'''--------------------------------Model----------------------------------'''

'''Positional Encoding'''

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
'''Transformer'''

class TransformerModel(nn.Module):

    def __init__(self, n_words, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
        self.model_type = 'Transformer'
        self.src_mask = None
        self.trg_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout) #get the positional encoder
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout) #get a layer
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) # get the encoder
        
        self.decoder = nn.Embedding(n_words, ninp) #decoder embedding
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.ninp = ninp
        self.out = nn.Linear(ninp, n_words)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, trg):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            src_device = src.device
            src_mask = self._generate_square_subsequent_mask(len(src)).to(src_device)
            self.src_mask = src_mask
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            trg_device = trg.device
            trg_mask = self._generate_square_subsequent_mask(len(trg)).to(trg_device)
            self.trg_mask = trg_mask
            
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        enc_output = self.transformer_encoder(src, self.src_mask)
        trg = self.decoder(trg) * math.sqrt(self.ninp)
        trg = self.pos_encoder(trg)
        dec_output = self.transformer_decoder(trg, enc_output, self.trg_mask)
        preds = self.out(dec_output)
        
        return preds
    
    
    
'''---------------------------------------------Train--------------------------'''

'''initiate'''
n_words = wordbank.n_words
ninp = 100 # embedding dimension
nhid = 100 # the dimension of the feedforward network model in nn.TransformerEncoder & nn.TransformerDecoder
nlayers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

model = TransformerModel(n_words, ninp, nhead, nhid, nlayers, dropout).to(device)

'''function to randomly choose'''

def chooseFromDataset(index):
    sample=positionsDataset[index]
    return (sample['input'],sample['target'])

n_iters = math.floor(0.9*lenTrainSet)
n_evaluate = math.floor(0.05*lenTrainSet)
n_test = math.floor(0.05*lenTrainSet)

training_pairs = [chooseFromDataset(i) for i in range(n_iters)]
evaluating_pairs = [chooseFromDataset(i) for i in range(n_iters, n_iters + n_evaluate)]


'''train() & evaluate)()'''
    
criterion = nn.CrossEntropyLoss()
lr = 0.0005 # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(epoch):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    n_words = wordbank.n_words
    
    for iter_time in range(1, n_iters + 1):
        training_pair = training_pairs[iter_time - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        trg_input = target_tensor[:-1,:]
        trg_output = target_tensor[1:,:]

        optimizer.zero_grad()
        preds = model(input_tensor, trg_input)

        loss = criterion(preds.view(-1,n_words), trg_output.reshape(-1))
        if epoch != epochs:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if iter_time % log_interval == 0 and iter_time > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} iters | '
                  'lr {:02.2f} | ms/iter {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, iter_time, n_iters, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    
        
def evaluate(eval_model, evaluating_pairs):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    n_words = wordbank.n_words

    with torch.no_grad():
        for iter_time in range(1, n_iters + 1):
            training_pair = training_pairs[iter_time - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            trg_input = target_tensor[:-1,:]
            trg_output = target_tensor[1:,:]

            preds = eval_model(input_tensor, trg_input)
            preds_flat = preds.view(-1,n_words)
            total_loss += criterion(preds_flat, trg_output.reshape(-1)).item()
    return total_loss / len(evaluating_pairs)

epochs = 3
def run(epochs=3):
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(epoch)
        val_loss = evaluate(model, evaluating_pairs)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
    
        scheduler.step()

    return best_model



print('='*80+'The Model is Training Now'+'='*80)
best_model = run()

'''testing'''

testing_pairs = [chooseFromDataset(i) for i in range(n_iters + n_evaluate, n_iters + n_evaluate + n_test)]
test_loss = evaluate(best_model, testing_pairs)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)



'''-------------------------------------print results-----------------------------'''

n_print = 10

def evaluateFromDataset(index):
    each_set=train_set[index]
    sample=positionsDataset[index]
    return (each_set['input'],each_set['target'],sample['input'],sample['target'])

printing_pairs=[evaluateFromDataset(random.randint(0,lenTrainSet-1)) for i in range(n_print)]

def generateTitle(best_model, input_tensor, max_len = 21):
    best_model.eval()
    n_words = wordbank.n_words
    
    with torch.no_grad():
        src_mask = best_model._generate_square_subsequent_mask(len(input_tensor))
        src = best_model.pos_encoder(input_tensor*math.sqrt(best_model.ninp))
        enc_output = best_model.transformer_encoder(src,src_mask)
        
        outputs = torch.zeros(max_len, dtype = torch.long)
        outputs[0] = torch.LongTensor([SOS_token])
#         print('enc:', enc_output)
        for i in range(1, max_len):
            trg_mask = best_model._generate_square_subsequent_mask(i)
            trg = best_model.pos_encoder(best_model.decoder(outputs[:i].unsqueeze(1))*math.sqrt(best_model.ninp))
#             print('trg:',trg)
            dec_output = best_model.transformer_decoder(trg, enc_output, trg_mask)
#             print('dec_out:', dec_output)
            dec_output = best_model.out(dec_output)
            out_flat = dec_output.view(-1,n_words)
            final = F.log_softmax(out_flat, dim=1)

            topv, topi = final[-1,:].data.topk(1)
            outputs[i] = topi.item()
            if topi.item() == EOS_token:
                break
                
    return ' '.join(wordbank.index2word[idx] for idx in outputs[:i].tolist())

def evaluateRandomly(best_model, n=10):
    for i in range(n):
        pair = printing_pairs[i]
        print('input sequence: ', pair[0])
        target_title = ' '.join(pair[1])
        print('target title: ', target_title)
        output_title = generateTitle(best_model, pair[2])
        print('output title: ', output_title)
        print('-'*80)
        
evaluateRandomly(best_model)
