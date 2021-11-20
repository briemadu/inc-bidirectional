#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 09:26:59 2020

@author: brie
"""
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from itertools import chain

from torchcrf import CRF
from transformers import (DistilBertModel, DistilBertPreTrainedModel, 
                          DistilBertForSequenceClassification,
                          BertForSequenceClassification,
                          BertForTokenClassification)

from utils import get_glove_embeddings

      
SIZE_PRED_EMBS = 5 # for True/False in SRL predicate

class vanillaLSTM(nn.Module):
    """
    A nn model, either LSTM or BiLSTM.
    Includes embedding layer, dropout, encoder and linear decoder.
    Loss is also defined here.
    """
    
    def __init__(self, size_vocab, size_embs, nhid, nlayers, nlabels, dropout, 
                 pad_symbol, corpus, no_glove=False, freeze=False,
                 bidirectional=False):
        super(vanillaLSTM, self).__init__()
        
        self.task = corpus.task
        self.label_pad_id = corpus.label2id['<pad>']
        if 'srl' in self.task:
            self.pred = corpus.label2id['B-V']
        # choose loss function (here to make output equivalent to CRF version)
        self.criterion = nn.CrossEntropyLoss()
        
        # embedding layer
        if no_glove:
            self.encoder = nn.Embedding(size_vocab, size_embs, 
                                        padding_idx=pad_symbol)
        else:
            pre_trained = get_glove_embeddings(size_vocab, size_embs, 
                                               corpus.word2id)
            self.encoder = nn.Embedding.from_pretrained(
                pre_trained, freeze=freeze, padding_idx=pad_symbol
                )
        if 'srl' in self.task:
            # create predicate embeddings (True of False)
            self.encoder_pred = nn.Embedding(2, SIZE_PRED_EMBS)
            size_embs = size_embs + SIZE_PRED_EMBS 
        # other layers        
        self.drop = nn.Dropout(dropout)
        if bidirectional:
            self.rnn = nn.LSTM(size_embs, nhid, nlayers, batch_first=True,
                               bidirectional=True)
            self.decoder = nn.Linear(nhid*2, nlabels)
        else:
            self.rnn = nn.LSTM(size_embs, nhid, nlayers, batch_first=True)
            self.decoder = nn.Linear(nhid, nlabels) 
     
    def forward(self, x, lens, y, seq2seq=True):
        """
        x is a padded tensor: dim=(batch, sequence length)
        lens is a tensor with true len of each sequence: dim=(batch)
        """
        
        # beware that initial padded x has seq size of longest sequence on 
        # corpus, final padded has seq size of longest sequence on batch
        
        # embed
        # (batch, sequence length) -> (batch, sequence length, embedding size)
        emb = self.encoder(x)
        # if SRL, embed predicate (True/False) and concatenate
        # (batch, sequence length, embedding size+pred embedding size)
        if 'srl' in self.task:
            predicates = (y==self.pred).type(torch.long) 
            emb_pred = self.encoder_pred(predicates)
            emb = torch.cat((emb, emb_pred), dim=2)    
        # apply dropout, no change in dimensions
        drop = self.drop(emb)
        # create batches without padding for lstm layer
        packed = pack_padded_sequence(drop, lens, batch_first=True, 
                                      enforce_sorted=False)
        # pass through lstm
        output, (hidden, context) = self.rnn(packed)
        # undo packing, padding is added again (all 0 vectors)
        # (batch, sequence length, hidden size)
        # sequence length is less than before: packing and unpacking removes
        # unnecessary padding, reduces to current batch max langth
        unpacked, newlens = pad_packed_sequence(output, batch_first=True)
        # another dropout layer, no change in dimensions
        drop2 = self.drop(unpacked)
        
        if seq2seq:
            # remove all pad vectors and pile real output vectors
            # (batch, max_batch_pad, hidden size) -> (sum(lens), hidden size)
            not_padded = torch.cat([drop2[i][:newlens[i]] 
                                    for i in range(unpacked.shape[0])])
            # decoded is already flattened to go into loss
            # (sum(lens), sequence length, #labels)
            decoded = self.decoder(not_padded)
            # get predictions for classes
            predicted = torch.argmax(F.log_softmax(decoded, dim=1), dim=1)
            # reshape y to fit into loss function
            y = y.view(-1)
            y = y[y != self.label_pad_id]
            # get loss
            loss = self.criterion(decoded, y)
            
        else:
            if self.rnn.bidirectional:
                # remove all pad vectors and pile real output vectors of last
                # timestep (stored in newlens, -1 because counts from 0)
                # (batch, max_batch_pad, hidden size) -> (batch, hidden size)
                # get last representation of forward lstm
                not_padded_f = drop2[range(drop2.shape[0]),newlens-1]
                # get last representation of backwards lstm
                not_padded_b = drop2[range(drop2.shape[0]),0]
                not_padded = not_padded_f + not_padded_b
            else:
                # remove all pad vectors and pile real output vectors of last
                # timestep (stored in newlens, -1 because counts from 0)
                # (batch, max_batch_pad, hidden size) -> (batch, hidden size)
                not_padded = drop2[range(drop2.shape[0]),newlens-1]
            # decoded is already flattened to go into loss
            # (batch, #labels)
            decoded = self.decoder(not_padded)
            # get predictions for classes
            predicted = torch.argmax(F.log_softmax(decoded, dim=1), dim=1)
            # reshape y to fit into loss function
            y = y.view(-1)
            # get loss
            loss = self.criterion(decoded, y)
        
        return loss, predicted

    
class LSTMCRF(nn.Module):
    """
    A nn model, either LSTM or BiLSTM with a CRF layer on top.
    Includes embedding layer, dropout, encoder, CRF and linear decoder.
    Loss is also defined here.
    """
        
    def __init__(self, size_vocab, size_embs, nhid, nlayers, nlabels, dropout, 
                 pad_symbol, corpus, no_glove=False, freeze=False,
                 bidirectional=False):
        super(LSTMCRF, self).__init__()
        
        self.task = corpus.task
        if 'srl' in self.task:
            self.pred = corpus.label2id['B-V']
        # embedding layer
        if no_glove:
            self.encoder = nn.Embedding(size_vocab, size_embs, 
                                        padding_idx=pad_symbol)
        else:
            pre_trained = get_glove_embeddings(size_vocab, size_embs, 
                                               corpus.word2id)
            self.encoder = nn.Embedding.from_pretrained(
                pre_trained, freeze=freeze, padding_idx=pad_symbol
                )
        if 'srl' in self.task:
            # create predicate embeddings (True of False)
            self.encoder_pred = nn.Embedding(2, SIZE_PRED_EMBS)
            size_embs = size_embs + SIZE_PRED_EMBS 
        # other layers
        self.drop = nn.Dropout(dropout)
        if bidirectional:
            self.rnn = nn.LSTM(size_embs, nhid, nlayers, batch_first=True,
                               bidirectional=True)
            self.decoder = nn.Linear(nhid*2, nlabels) 
        else:
            self.rnn = nn.LSTM(size_embs, nhid, nlayers, batch_first=True)
            self.decoder = nn.Linear(nhid, nlabels) 
        self.crf = CRF(nlabels, batch_first=True)
     
    # seq2seq below just to be consistent with train function, will not be used
    def forward(self, x, lens, y, seq2seq=None):
        """
        x is a padded tensor: dim=(batch, sequence length)
        lens is a tensor with true len of each sequence: dim=(batch)
        """
        # beware that initial padded x has seq size of longest sequence on 
        # corpus, final padded has seq size of longest sequence on batch
        
        # embed
        # (batch, sequence length) -> (batch, sequence length, embedding size)
        emb = self.encoder(x)
        # if SRL, embed predicate (True/False) and concatenate
        # (batch, sequence length, embedding size+pred embedding size)
        if 'srl' in self.task:
            predicates = (y==self.pred).type(torch.long) 
            emb_pred = self.encoder_pred(predicates)
            emb = torch.cat((emb, emb_pred), dim=2)
        # apply dropout, no change in dimensions
        drop = self.drop(emb)
        # create batches without padding for lstm layer
        packed = pack_padded_sequence(drop, lens, batch_first=True, 
                                      enforce_sorted=False)
        # pass through lstm
        output, (hidden, context) = self.rnn(packed)
        # undo packing, padding is added again (all 0 vectors)
        # (batch, sequence length, hidden size)
        # sequence length is less than before: packing and unpacking removes
        # unnecessary padding, reduces to current batch max langth
        unpacked, newlens = pad_packed_sequence(output, batch_first=True)
        # another dropout layer, no change in dimensions
        drop2 = self.drop(unpacked)      
        # (batch, sequence length, #labels)
        decoded = self.decoder(drop2)
        # truncate extra padding for this batch in y
        y = y[:,:unpacked.shape[1]] 
        # mask=False if symbol is pad, use y as reference 
        # (pad = num_tags-1 because tags start at 0)
        mask = y.lt(self.crf.num_tags).type(torch.uint8)        
        # crf module needs y not to have an extra symbol for the padding, 
        # otherwise it throws error (it tries to access such index in the 
        # transition matrix). So I replace the padding symbol by the 
        # greatest tag id (could be any other tag), because the mask will make 
        # it be ignored anyway.
        y_ = y - (y > self.crf.num_tags-1).type(torch.uint8)
        # get loss
        loglikelihood = self.crf(decoded, y_, mask)        
        # get predictions for classes
        best_seqs = self.crf.decode(decoded, mask)
        best_seqs = torch.tensor(list(chain.from_iterable(best_seqs)), 
                                 device=y.device)
        
        # return negative of loglikelihood because we use minimization
        return -loglikelihood, best_seqs
    
    
class BERT(nn.Module):
    """
    Pretrained BERT Model for sequence classification.
    Loss is included here.
    """
    
    def __init__(self, nlabels, corpus, seq2seq, bert_model, dropout):
        super(BERT, self).__init__()
        
        self.task = corpus.task
        self.bert_model = bert_model
        self.wordpad = corpus.tokenizer.pad_token_id
        self.dummylabel = corpus.label2id['<pad>']
        
        if 'distil' in bert_model:
            if seq2seq:
                self.encoder = DistilBertForTokenClassification.from_pretrained(
                        bert_model,
                        finetuning_task=corpus.task,
                        num_labels=nlabels) 
            else:
                self.encoder = DistilBertForSequenceClassification.from_pretrained(
                   bert_model,
                   finetuning_task=corpus.task,
                   num_labels=nlabels) # -1 to ignore dummy variable that is not used here
        else:
            if seq2seq:
                self.encoder = BertForTokenClassification.from_pretrained(
                        bert_model,
                        finetuning_task=corpus.task,
                        num_labels=nlabels) 
            else:
                self.encoder = BertForSequenceClassification.from_pretrained(
                   bert_model,
                   finetuning_task=corpus.task,
                   num_labels=nlabels)
                
        # choose loss function (here to make output equivalent to CRF version)
        self.criterion = nn.CrossEntropyLoss()
        
        if 'srl' in self.task:
            # create predicate embeddings (True of False)
            #self.encoder_pred = nn.Embedding(2, nlabels)
            # use token_type_ids
            self.pred = corpus.label2id['B-V']
        

    def forward(self, x, lens, y, seq2seq=None):
        
        # x: (batch, seq_len)
        # y: (batch, seq_len)
        # mask: (batch, seq_len)
        mask = (x != self.wordpad).type(torch.uint8)
        
        # because according to the documentation token_type_ids is an embedding
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # we can 'misuse' it to be a flag for predicate instead of a flag for
        # first/second sencente
        if 'srl' in self.task:
            preds = (y==self.pred).type(torch.long)
        
            # scores: (batch, seq_len, n_labels=valid labels + dummy/pad label)
            scores = self.encoder(x, attention_mask=mask, token_type_ids=preds)[0]
        else:
            scores = self.encoder(x, attention_mask=mask)[0]

        if seq2seq:
            
            valid_labels_mask = (y != self.dummylabel)
            # flatten scores that correspond to original tokens only
            # (sum_valid_lens, n_labels)
            valid_scores = scores[valid_labels_mask]
            # flatten labels that are valid
            # (sum_valid_lens)
            valid_labels = y[valid_labels_mask]
            
            loss = self.criterion(valid_scores, valid_labels)
            
            # (sum_valid_lens, n_labels) -> (sum_valid_lens)
            predicted = torch.argmax(F.log_softmax(valid_scores, dim=1), dim=1)
        
        else:
            valid_y = y[:,1] # sequence label is for sure at position 1 (0 is dummy for CLS)
            loss = self.criterion(scores, valid_y)
            predicted = torch.argmax(F.log_softmax(scores, dim=1), dim=1)
              
        return loss, predicted
    
    
class DistilBertForTokenClassification(DistilBertPreTrainedModel):
    
    """
    My adaptation for DistilBERT, copying from BertForTokenClassification
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, head_mask=None, 
                inputs_embeds=None, labels=None):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), 
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
    
