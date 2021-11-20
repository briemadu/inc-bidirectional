#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:22:26 2020

@author: brie
"""
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from numpy import asarray
import numpy as np
import nltk

from seqeval.metrics import f1_score, precision_score, recall_score


NP_SEED = 2204


def train(loader, model, my_device, optimizer, clip, label_pad, seq2seq):
    """
    Perform a training epoch.
    
    Parameters
    ----------
    loader : torch.utils.data.dataloader.DataLoader
        Training data loader.
    model : models.<model>
        One of the models in the models module.
    my_device : torch.device
        A PyTorch device.
    optimizer : torch.optim.<optimizer>
        A PyTorch optimizer.
    clip : float
        Clipping size.
    label_pad : int
        Index of padding symbol in the labels dictionary.
    seq2seq : bool
        True if sequence tagging, else False for sequence classification.

    Returns
    -------
    total_loss : float
        Loss value.
    all_predictions : numpy.ndarray
        Predicted labels for all tokens (or sentences if seq2label)
    all_labels : numpy.ndarray
        Respective gold labels for all tokens (or sentences if seq2label),
        needed because order changes due to shuffling.

    """

    model.train()
    total_loss = 0
    all_predictions = np.array([])
    all_labels = np.array([])
    # loop over batches
    for x, lens, y, idx in loader:
        
        x = x.to(my_device)
        y = y.to(my_device)
        lens = lens.to(my_device)
        
        loss, predicted = model(x, lens, y, seq2seq) 

        total_loss += loss.item()
        
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
        optimizer.step()
        optimizer.zero_grad()
        
        # had to add this if because of marks for subtokens that incremental results need
        if hasattr(model, 'bert_model') and not seq2seq:
            y = y[:,1]
        else:    
            y = y.reshape(-1)
            y = y[y != label_pad] # ignore padding
        
        all_predictions = np.append(all_predictions, predicted.cpu().numpy())
        all_labels = np.append(all_labels, y.cpu().numpy())
    
    print(' TRAINING \t Loss: {:.2f}'.format(total_loss))     
        
    return total_loss, all_predictions, all_labels 


def test(loader, model, my_device, label_pad, seq2seq, dataset='validation'):
    """
    Perofrm test.

    Parameters
    ----------
    loader : torch.utils.data.dataloader.DataLoader
        Training data loader.
    model : models.<model>
        One of the models in the models module.
    my_device : torch.device
        A PyTorch device.
    label_pad : int
        Index of padding symbol in the labels dictionary.
    seq2seq : bool
        True if sequence tagging, else False for sequence classification.
    dataset : str
        'validation' or 'test', used for printing results.

    Returns
    -------
    total_loss : float
        Loss value.
    all_predictions : numpy.ndarray
        Predicted labels for all tokens (or sentences if seq2label)
    all_labels : numpy.ndarray
        Respective gold labels for all tokens (or sentences if seq2label)
        needed because order changes due to shuffling.

    """
     
    model.eval()
    total_loss = 0
    all_predictions = np.array([])
    all_labels = np.array([])
    
    with torch.no_grad():
        # loop over batches
        for x, lens, y, idx in loader:
            
            x = x.to(my_device)
            y = y.to(my_device)
            lens = lens.to(my_device)
        
            loss, predicted = model(x, lens, y, seq2seq) 
            
            total_loss += loss.item()
            
            if hasattr(model, 'bert_model') and not seq2seq:
                 y = y[:,1]
            else:
                y = y.reshape(-1)
                y = y[y != label_pad] # ignore padding
            
            all_predictions = np.append(all_predictions, predicted.cpu().numpy())
            all_labels = np.append(all_labels, y.cpu().numpy())
            
    print(' '+dataset.upper()+' \t Loss: {:.2f}'.format(total_loss))   
    
    return total_loss, all_predictions, all_labels


def load_data(corpus, batch_size=64, test_batch_size=1, sample=False, model=""):
    """
    Create train, validation and test data loaders.

    Parameters
    ----------
    corpus : structs.Corpus
        The corpus.
    batch_size : int, optional
        Size of batch for training/validation. The default is 64.
    test_batch_size : int, optional
        Size of batch for test. The default is 1.
    sample : bool, optional
        Whether to do truncated training. The default is False.
        Truncated training samples a length for each sequence, less than or
        equal to the original length, and truncated the sequence.
    model : str, optional
        Name of the model. The default is "".

    Returns
    -------
    train_loader : torch.utils.data.dataloader.DataLoader
        Data loader for the training set.
    valid_loader : torch.utils.data.dataloader.DataLoader
        Data loader for the validation set.
    test_loader : torch.utils.data.dataloader.DataLoader
        Data loader for the test set.

    """

    # first pad sequences, than create data loader
    train_dataset = pad_seqs(corpus, corpus.train, sample, model)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=True)

    valid_dataset = pad_seqs(corpus, corpus.valid, sample, model)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, 
                              shuffle=True)
    
    test_dataset = pad_seqs(corpus, corpus.test) # no sampling
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, 
                             shuffle=False)
    
    
    return train_loader, valid_loader, test_loader

def pad_seqs(corpus, data, sample=False, model=""):
    """
    Pad sequences.

    Parameters
    ----------
    corpus : structs.Corpus
        The corpus.
    data : dict
        Dictionary mapping from sequence index to tuple of ordered token
        labels.
    sample : bool, optional
        Whether to do truncated training. The default is False.
        Truncated training samples a length for each sequence, less than or
        equal to the original length, and truncated the sequence.
    model : str, optional
        Name of the model. The default is "".

    Returns
    -------
    dataset : torch.utils.data.dataset.TensorDataset
        PyTorch's tensor dataset with padded sequences. It loads the sequence,
        their lengths, the padded gold labels and the sequence indexes.

    """
    
    np.random.seed(NP_SEED)
    
    # truncated training, we sample a length and stripp off the ending of
    # that sequence
    if sample:
        # BERT has to start at 2 otherwise there could be sentences with only
        # the invalid CLS token, throws error at cross entropy function.
        if 'bert' in model and corpus.task in ['sentiment', 'atis_intent', 
                                               'snips_intent', 'objsubj',
                                               'proscons', 'posneg', 
                                               'sent_negpos']:
            sample_lens = [np.random.randint(2, len(corpus.id2seq[i])+1) 
                           for i in data.keys()]
        else:
            sample_lens = [np.random.randint(1, len(corpus.id2seq[i])+1) 
                           for i in data.keys()]
        
        padded_seqs = pad_sequence([torch.tensor(corpus.id2seq[i][:sample_lens[n]]) 
                                    for n, i in enumerate(data.keys())], 
                                   batch_first=True, 
                                   padding_value=corpus.word2id['<pad>'])
        
        len_seqs = torch.tensor(sample_lens)
        
        padded_labels =  pad_sequence([torch.tensor(data[i][:sample_lens[n]]) 
                                       for n, i in enumerate(data.keys())], 
                                      batch_first=True, 
                                      padding_value=corpus.label2id['<pad>'])
        
    
    else:    
        padded_seqs = pad_sequence([torch.tensor(corpus.id2seq[i]) 
                                    for i in data.keys()], 
                                   batch_first=True, 
                                   padding_value=corpus.word2id['<pad>'])
        
        len_seqs = torch.tensor([len(corpus.id2seq[i]) for i in data.keys()])
        
        padded_labels =  pad_sequence([torch.tensor(data[i]) 
                                       for i in data.keys()], 
                                      batch_first=True, 
                                      padding_value=corpus.label2id['<pad>'])
    
    indexes = torch.tensor([i for i in data.keys()])
    
    dataset = TensorDataset(padded_seqs, len_seqs, padded_labels, indexes)
    
    return dataset


def load_glove_embeddings(dim, vocab):
    """
    Load GloVe embedding vectors for all words in our vocabulary.
    https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

    Parameters
    ----------
    dim : int
        Dimension of GloVe embeddings. Can be 50, 100, 200 and 300.
    vocab : dict
        Dictionary mapping words to index.

    Returns
    -------
    embeddings_index : dict
        A dictionary that maps word to embedding vector.
    
    """

    embeddings_index = dict()
    lower_dict = [word.lower() for word in vocab.keys()]
    
    with open('glove.6B/glove.6B.'+str(dim)+'d.txt', 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
            # use only low case? GloVe seems to use only low case, but what about NER?
            if word in vocab:
                embeddings_index[vocab[word]] = coefs
            # maybe Word get same embedding as word? 
            elif word in lower_dict:
                try:
                    embeddings_index[vocab[word.title()]] = coefs
                except KeyError:
                    continue
                   
    return embeddings_index


def get_glove_embeddings(ntoken, embsize, vocab):
    """
    
    Create embedding matrix used in the models.
    If a word was not in GloVe, initialize it randomly (ToDo: improve it)

    Parameters
    ----------
    ntoken : int
        Number of words in the vocabulary.
    embsize : int
        Embedding size.
    vocab : dict
        Dictionary mapping words to index.

    Returns
    -------
    torch.tensor
        Embedding matrix with dims ntoken x embedding dimension

    """
    
    glove_embeddings = load_glove_embeddings(embsize, vocab)
    
    emb_matrix = np.empty(shape=(ntoken, embsize))
    for idx in range(ntoken):
        if idx in glove_embeddings:
            emb_matrix[idx] = glove_embeddings[idx]
        else:
            emb_matrix[idx] = np.random.normal(size=embsize)
           
    return torch.FloatTensor(emb_matrix)


def evaluate_nn(gold_labels, predictions, id2label):
    """

    Parameters
    ----------
    gold_labels : numpy.ndarray
        Predicted labels for all tokens (or sentences if seq2label)
    predictions : numpy.ndarray
        Respective gold labels for all tokens (or sentences if seq2label)
    id2label : dict
        Dictionary that maps from ids to real labels.

    Returns
    -------
    acc : float
        Accuracy of predicted labels.
    f1_e : float
        F! Score of predicated labels, adapted for BIO scheme (seqeval module)

    """
    
    acc = (predictions == gold_labels).sum() /len(predictions)
    
    # token level
    #precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
    #                              gold_labels, predictions, average='micro')

    # seqeval needs IOB2 labels, convert it
    preds = [id2label[x] for x in predictions]
    golds = [id2label[x] for x in gold_labels]
    
    # entity level
    precision_e = precision_score(golds, preds)
    recall_e = recall_score(golds, preds)
    f1_e = f1_score(golds, preds)
    
    print('         \t Accuracy: {:.3f}'.format(acc))
    #print('  sklearn   \t Prec: {:.2f} \t Recall: {:.2f} \t F1: {:.2f}'.format(
    #    precision, recall, f1))
    print('         \t Precision: {:.2f} \t Recall: {:.2f} \t F1: {:.2f}'.format(
        precision_e, recall_e, f1_e))
   
    return acc, f1_e


def get_partial_outputs(loader, model, my_device, label_pad, seq2seq):
    """
    Get incremental outputs (no prophecies).

    Parameters
    ----------
    loader : torch.utils.data.dataloader.DataLoader
        Data loader, batch must have size 1.
    model : models.<model>
        NN model not BERT.
    my_device : torch.device
        PyTorch device.
    label_pad : int
        Index of padding label.
    seq2seq : bool
        True if sequence tagging, else False for sequence classification.

    Returns
    -------
    results : dict of dicts
        A dictionary storing partial outputs, accuracy w.r.t. the gold labels
        and an np matrix that indicates editions.

    """

    model.eval()
    
    results = {'partial_outputs':{}, 'log_changes':{}, 'accuracy':{}} 
    
    with torch.no_grad():
        
        for x, lens, y, idx in loader:
            
            x = x.to(my_device)
            y = y.to(my_device)
            lens = lens.to(my_device)
            
            if seq2seq:
                # lower diagonal matrix to store increasing prefixes of 
                # the output
                predictions = np.zeros((lens, lens))
                # lower diagonal matrix to store edits (1 if that label)
                # was edited in comparison to last partial output
                changes = np.zeros((lens, lens))
            else:
                predictions = np.zeros((lens, 1))
                changes = np.zeros((lens, 1))
            
            # we loop over successively increasing prefixes of the input 
            # sequence
            for l in range(1,lens.item()+1):
           
                part_x = x[:,:l]
                if seq2seq:
                    part_y = y[:,:l]
                else:
                    part_y = y
                part_len = torch.tensor([l])
                # predict labels of partial input
                _, predicted = model(part_x, part_len, part_y, seq2seq)
                
                if seq2seq:
                    predictions[l-1] = np.array((predicted.tolist()
                                             + (lens.item() - l)*[np.inf]))
                else:
                    predictions[l-1] = np.array((predicted.tolist()))
                
                if l == 1:
                    changes[l-1][0] = 1
                else:
                    changes[l-1] = predictions[l-1] != predictions[l-2]
            
            y = y.reshape(-1)
            y = torch.tensor([i for i in y if i!=label_pad])
            
            if seq2seq:
                acc = (predictions[-1] == y.numpy()).sum() / lens.item()
            else:
                acc = (predictions[-1] == y.numpy()).sum()
            
            results['partial_outputs'][idx.item()] = predictions
            results['log_changes'][idx.item()] = changes
            results['accuracy'][idx.item()] = acc
            
    return results

def get_bert_partial_outputs(loader, model, my_device, label_pad, word_pad, seq2seq):
    """
    Parameters
    ----------
    loader : torch.utils.data.dataloader.DataLoader
        Data loader, batch must have size 1.
    model : models.<model>
        NN BERT model.
    my_device : torch.device
        PyTorch device.
    label_pad : int
        Index of padding label.
    seq2seq : bool
        True if sequence tagging, else False for sequence classification.

    Returns
    -------
    results : dict of dicts
        A dictionary storing partial outputs, accuracy w.r.t. the gold labels
        and an np matrix that indicates editions.

    """
    # see comments in function above
    model.eval()
    
    results = {'partial_outputs':{}, 'log_changes':{}, 'accuracy':{}} 
    
    with torch.no_grad():
        
        for x, lens, y, idx in loader:
            
            x = x.to(my_device)
            y = y.to(my_device)
            
            # ignore subtokens that are not in first position (BERT tokenization)
            valid_tokens =  np.argwhere((y[0]!=label_pad).cpu())[0].tolist()
            lens = len(valid_tokens)
            
            if seq2seq:
                predictions = np.zeros((lens, lens))
                changes = np.zeros((lens, lens))
            else:
                predictions = np.zeros((lens, 1))
                changes = np.zeros((lens, 1))
                
            for l in range(1,lens+1):
                                
                last_token = valid_tokens[l-1]
                part_x = x[:,:last_token+1]
                part_y = y[:,:last_token+1]
                
                # last label include up to last subtoken   
                if l == lens:
                    part_x = x[x != word_pad].unsqueeze(0)
                    part_y = y[:,:part_x.shape[1]]

                _, predicted = model(part_x, None, part_y, seq2seq)
                
                if seq2seq:
                    predictions[l-1] = np.array((predicted.tolist()
                                             + (lens - l)*[np.inf]))
                else:
                    predictions[l-1] = np.array((predicted.tolist()))
                
                if l == 1:
                    changes[l-1][0] = 1
                else:
                    changes[l-1] = predictions[l-1] != predictions[l-2]
            
            y = y[y!=label_pad]
            
            if seq2seq:
                acc = (predictions[-1] == y.cpu().numpy()).sum() / lens
            else:
                acc = (predictions[-1] == y[0].cpu().numpy()).sum()
            
            results['partial_outputs'][idx.item()] = predictions
            results['log_changes'][idx.item()] = changes
            results['accuracy'][idx.item()] = acc
            
    return results


def get_partial_outputs_with_prophecies(prophecies, loader, model, my_device, 
                                        corpus, seq2seq):
    """
    

    Parameters
    ----------
    prophecies : dict
        Dictionary mapping from sequence index to a list of prophecies, one
        for each prefix in the sequence.
    loader : torch.utils.data.dataloader.DataLoader
        Data loader, batch must have size 1.
    model : models.<model>
        NN model not BERT
    my_device : torch.device
        PyTorch device.
    label_pad : int
        Index of padding label.
    seq2seq : bool
        True if sequence tagging, else False for sequence classification.

    Returns
    -------
    results : dict of dicts
        A dictionary storing partial outputs, accuracy w.r.t. the gold labels
        and an np matrix that indicates editions.

    """
    # see comments in function above
    model.eval()
    
    results = {'partial_outputs':{}, 'log_changes':{}, 'accuracy':{}} 
    
    with torch.no_grad():
        
        for x, lens, y, idx in loader:
            
            #if idx.item() not in prophecies:
            #    continue
                
            x = x.to(my_device)
            y = y.to(my_device)
            lens = lens.to(my_device)
            
            if seq2seq:
                predictions = np.zeros((lens, lens))
                changes = np.zeros((lens, lens))
            else:
                predictions = np.zeros((lens, 1))
                changes = np.zeros((lens, 1))
            
            pad = corpus.word2id['<pad>']
            
            for l in range(1,lens.item()+1):
                
                if l != lens.item():
                    part_x = x[:,:l]
                    # add prophecy
                    prophecy = nltk.word_tokenize(
                        prophecies[idx.item()][l-1][0])
                    prophecy_ids = torch.tensor([[corpus.word2id.get(w, pad) 
                                                  for w in prophecy[l:]]], 
                                            dtype=torch.long, device=x.device)
                    
                    part_x = torch.cat((part_x, prophecy_ids),dim=1) 
                    part_len = torch.tensor([l+prophecy_ids.shape[1]], 
                                            device=x.device)
                    # create any y to append will not be used (but cannot be the same idx as
                    # label of predicate in SRL), we use zero and check
                    if 'srl' in corpus.task:
                        assert corpus.label2id['B-V'] != 0
                    if seq2seq:
                        extra_pad = torch.tensor([[0]*(part_x.shape[1]-l)], device=x.device, dtype=torch.long)
                        part_y = torch.cat((y[:,:l], extra_pad), dim=1)
                        #part_y = torch.zeros((1, part_len.item()), dtype=torch.long, 
                        #                device=y.device)
                    else:
                        part_y = y
                else: # complete sentence does not need prophecy
                    part_x = x
                    part_y = y
                    part_len = lens
                
                #unpacked, mask = model(x, lens) # _ = (hidden, context)
                _, predicted = model(part_x, part_len, part_y, seq2seq)
                
                if seq2seq:
                    predictions[l-1] = np.array((predicted[:l].tolist()
                                             + (lens.item() - l)*[np.inf]))
                else:
                    predictions[l-1] = np.array((predicted.tolist()))
                
                if l == 1:
                    changes[l-1][0] = 1
                else:
                    changes[l-1] = predictions[l-1] != predictions[l-2]
                
            y = y.reshape(-1)
            y = torch.tensor([i for i in y  if i!=corpus.label2id['<pad>']])
            
            if seq2seq:
                acc = (predictions[-1] == y.cpu().numpy()).sum() / lens.item()
            else:
                acc = (predictions[-1] == y.cpu().numpy()).sum()
            
            results['partial_outputs'][idx.item()] = predictions
            results['log_changes'][idx.item()] = changes
            results['accuracy'][idx.item()] = acc

        
    return results

def get_bert_partial_outputs_with_prophecies(prophecies, loader, model, my_device, 
                                        corpus, seq2seq):
    """
    

    Parameters
    ----------
    prophecies : dict
        Dictionary mapping from sequence index to a list of prophecies, one
        for each prefix in the sequence.
    loader : torch.utils.data.dataloader.DataLoader
        Data loader, batch must have size 1.
    model : models.<model>
        NN BERT model.
    my_device : torch.device
        PyTorch device.
    label_pad : int
        Index of padding label.
    seq2seq : bool
        True if sequence tagging, else False for sequence classification.

    Returns
    -------
    results : dict of dicts
        A dictionary storing partial outputs, accuracy w.r.t. the gold labels
        and an np matrix that indicates editions.

    """
    # see comments in function above
    model.eval()
    
    results = {'partial_outputs':{}, 'log_changes':{}, 'accuracy':{}} 
    
    label_pad = corpus.label2id['<pad>']
    word_pad = corpus.word2id['<pad>']
    
    with torch.no_grad():
        
        for x, lens, y, idx in loader:
                
            x = x.to(my_device)
            y = y.to(my_device)
            
            
            valid_tokens =  np.argwhere((y[0]!=label_pad).cpu())[0].tolist()
            lens = len(valid_tokens)
            
            if seq2seq:
                predictions = np.zeros((lens, lens))
                changes = np.zeros((lens, lens))
            else:
                predictions = np.zeros((lens, 1))
                changes = np.zeros((lens, 1))
            
            for l in range(1,lens+1):
                
                if l != lens:

                    last_token = valid_tokens[l-1]
                    part_x = x[:,:last_token+1]
                    part_y = y[:,:last_token+1]
                    
                    prophecy = corpus.tokenizer.encode(prophecies[idx.item()][l-1][0])
                    prophecy_ids = torch.tensor([prophecy], device=x.device)
                    
                    part_x = torch.cat((part_x, prophecy_ids),dim=1) 
                    #part_len = torch.tensor([part_y.shape[1]+prophecy_ids.shape[1]], 
                    #                        device=x.device)
                    
                    #create y
                    extra_pad = torch.tensor([[label_pad]*prophecy_ids.shape[1]], device=x.device)
                    part_y = torch.cat((part_y, extra_pad), dim=1)
                    
                    #part_y = torch.ones((1, part_len.item()), dtype=torch.long, 
                    #                    device=y.device)

                else: # complete sentence does not need prophecy
                    
                    part_x = x[x != word_pad].unsqueeze(0)
                    part_y = y[:,:part_x.shape[1]]

                
                #unpacked, mask = model(x, lens) # _ = (hidden, context)
                _, predicted = model(part_x, None, part_y, seq2seq)
                
                if seq2seq:
                    predictions[l-1] = np.array((predicted[:l].tolist()
                                             + (lens - l)*[np.inf]))
                else:
                    predictions[l-1] = np.array((predicted.tolist()))
                
                if l == 1:
                    changes[l-1][0] = 1
                else:
                    changes[l-1] = predictions[l-1] != predictions[l-2]
                
            y = y[y!=label_pad]
            
            if seq2seq:
                acc = (predictions[-1] == y.cpu().numpy()).sum() / lens
            else:
                acc = (predictions[-1] == y[0].cpu().numpy()).sum()
            
            results['partial_outputs'][idx.item()] = predictions
            results['log_changes'][idx.item()] = changes
            results['accuracy'][idx.item()] = acc

        
    return results


 
class voidExperiment:
    """
    A pseudo-experiment class, to have consistency in the code when comet_ml
    is not used. Created because we need e.g. "with experiment.train()" modes.
    """
    def __init__(self):
        pass
    def train(self, mode=True):
        self.training = mode
        return self
    def validate(self, mode=True):
        self.training = mode
        return self
    def test(self, mode=True):
        self.training = mode
        return self    
    def __enter__(self): 
        pass
    def __exit__(self, type, value, traceback):
        pass
    