#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:23:08 2020

@author: brie
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import transformers

from utils import  (get_partial_outputs, get_partial_outputs_with_prophecies,
                    get_bert_partial_outputs, get_bert_partial_outputs_with_prophecies)
                    
NP_SEED = 2204

  
class Corpus():
    """
    Read datasets and creates corpus object.
    """

    def __init__(self, task, seq2seq, max_len=60, P=0.98, no_unk=False):
        
        print('Reading corpus...')
        # sentences longer than that are not included
        self.max_len=max_len
        # (1-P) probability of a word being masked as UNK in training set 
        self.P = P
        self.task = task
        self.word2id = {}  # stores words
        self.word2id['<UNK>'] = len(self.word2id) 
        self.label2id = {}  # stores labels
        self.id2seq = {}  # stores sentences
        
        if seq2seq:
            self.train = self.build_seq2seq('data/train/train.'+task, train=True)
            self.valid = self.build_seq2seq('data/valid/valid.'+task, train=no_unk)
            self.test = self.build_seq2seq('data/test/test.'+task, train=no_unk)
        else:
            self.train = self.build_seq2label('data/train/train.'+task, train=True)
            self.valid = self.build_seq2label('data/valid/valid.'+task, train=no_unk)
            self.test = self.build_seq2label('data/test/test.'+task, train=no_unk)

        # pad symbol = # words (or # labels), because we count from 0
        self.word2id['<pad>'] = len(self.word2id) 
        self.label2id['<pad>'] = len(self.label2id)  
        
    
    def build_seq2seq(self, file, train=False):
        """
        Build corpus object for sequence tagging.

        Parameters
        ----------
        file : str
            Path to data file. It has to follow the scheme token \t label \n
            with an extra /n between sequences.
        train : bool, optional
            True if building the training set (so it randomly mask words and
            add new words to the vocabulary). The default is False.

        Returns
        -------
        data : dict
            Dictionary mapping sequence index to tuple of oredered label 
            indexes.

        """
        np.random.seed(NP_SEED)
        data = {}
    
        with open(file, 'r') as file:
            
            sentence = []
            tags = []
                
            for line in file:
       
                if line != '\n':
                    
                    word, label = line.split()
                    # only words in training set goes to vocab, 
                    # new words in validation or test set are unk
                    if train:
                        # mask some words in training set as UNK randomly    
                        if np.random.uniform(0,1) > self.P:
                            word = '<UNK>'
                        if word not in self.word2id:
                            self.word2id[word] = len(self.word2id)
                    # labels not seen in training set will also go to vocab
                    # to avoid errors (it may happen in IOB2 scheme that
                    # a B-tag was seen in training but an I-tag was not)
                    if label not in self.label2id:
                            self.label2id[label] = len(self.label2id)
                                   
                    sentence.append(self.word2id.get(word, 
                                                     self.word2id['<UNK>']))
                    tags.append(self.label2id[label])

                else:
                    # ignore outliers (too long sequences)
                    # although their words were included in vocab... remove?
                    if len(sentence) > self.max_len:
                        pass
                    else:
                        self.id2seq[len(self.id2seq)] = tuple(sentence)
                        data[len(self.id2seq)-1] = tuple(tags)
                        
                    sentence = []
                    tags = []
                    
        return data 
    
    def build_seq2label(self, file, train=False):
        """
        Build corpus object for sequence classification.

        Parameters
        ----------
        file : str
            Path to data file. It has to follow the scheme token \t label \n
            with an extra /n between sequences.
        train : bool, optional
            True if building the training set (so it randomly mask words and
            add new words to the vocabulary). The default is False.

        Returns
        -------
        data : dict
            Dictionary mapping sequence index to tuple of oredered label 
            indexes.

        """
        np.random.seed(NP_SEED)
        data = {}
    
        with open(file, 'r') as file:
            
            sentence = []
                
            for line in file:
                
                if line[:8] == '<LABEL>:':
                    
                    sent_label = line.split()[1]
                    if sent_label not in self.label2id:
                            self.label2id[sent_label] = len(self.label2id)
                
                elif line != '\n':
                    
                    word = line.split()[0]
                    # only words in training set goes to vocab, 
                    # new words in validation or test set are unk
                    if train:
                        # mask some words in training set as UNK randomly    
                        if np.random.uniform(0,1) > self.P:
                            word = '<UNK>'
                        if word not in self.word2id:
                            self.word2id[word] = len(self.word2id)
                              
                    sentence.append(self.word2id.get(word, 
                                                     self.word2id['<UNK>']))

                else:
                    # ignore outliers (too long sequences)
                    # although their words were included in vocab... remove?
                    if len(sentence) > self.max_len:
                        pass
                    else:
                        self.id2seq[len(self.id2seq)] = tuple(sentence)
                        data[len(self.id2seq)-1] = (self.label2id[sent_label],)
                        
                    sentence = []
                    
        return data 

        
class Corpus4Bert():
    """
    Read datasets and creates corpus object, using BERT's tokenization.
    """

    def __init__(self, bert_model, task, seq2seq, max_len=60, first_token=True):
        
        print('Reading corpus...')
        
        self.max_len = max_len
        self.bert_model = bert_model
        if 'distil' in bert_model:
            self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(bert_model)
        else:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(bert_model)

        self.task = task
        self.word2id = {}  # stores words
        self.label2id = {}  # stores labels
        self.label2id = {'<pad>':-1}  # dummy label for #tokens, wordpieces
        self.id2seq = {}  # stores sentences
        
        if seq2seq:
            self.first_token = first_token
            self.train = self.build_seq2seq('data/train/train.'+task)
            self.valid = self.build_seq2seq('data/valid/valid.'+task)
            self.test = self.build_seq2seq('data/test/test.'+task)
        else:
            self.train = self.build_seq2label('data/train/train.'+task)
            self.valid = self.build_seq2label('data/valid/valid.'+task)
            self.test = self.build_seq2label('data/test/test.'+task)

        # pad symbol = # words (or # labels), because we count from 0
        self.word2id['<pad>'] = self.tokenizer.pad_token_id 

        
    
    def build_seq2seq(self, file):

        data = {}
        CLS = [self.tokenizer.cls_token_id]
        SEP = [self.tokenizer.sep_token_id]
        DUMMY = [self.label2id['<pad>']]
    
        with open(file, 'r') as file:
            
            sentence = []
            tags = []
            len_seq = 0
                
            for line in file:
       
                if line != '\n':
                    
                    word, label = line.split()

                    if label not in self.label2id:
                        self.label2id[label] = len(self.label2id)-1
                    
                    tokens = self.tokenizer.encode(word)[1:-1]
                    sentence += tokens
                    aux_tags = [self.label2id[label]] + DUMMY*(len(tokens)-1)
                    # we can tag either the first token or the last in case
                    # of wordpiece segmentation
                    if not self.first_token and len(tokens)>1:
                        tags += aux_tags[::-1]     
                    else:
                         tags += aux_tags
                    len_seq += 1

                else:
                    # ignore outliers (too long sequences)
                    # although their words were included in vocab... remove?
                    if len_seq > self.max_len:
                        pass
                    else:
                        sentence = CLS + sentence + SEP
                        tags = DUMMY + tags + DUMMY
                        self.id2seq[len(self.id2seq)] = tuple(sentence)
                        data[len(self.id2seq)-1] = tuple(tags)
                        
                    sentence = []
                    tags = []
                    len_seq = 0
                    
        return data 
    
    def build_seq2label(self, file, train=False):
        
        data = {}
        CLS = [self.tokenizer.cls_token_id]
        SEP = [self.tokenizer.sep_token_id]
        DUMMY = [self.label2id['<pad>']]
    
        with open(file, 'r') as file:
            
            sentence = []
            tags = []
            len_seq = 0
                
            for line in file:
                
                if line[:8] == '<LABEL>:':
                    
                    sent_label = line.split()[1]
                    if sent_label not in self.label2id:
                            self.label2id[sent_label] = len(self.label2id)-1
                
                elif line != '\n':
                    
                    word = line.split()[0]
                    
                    tokens = self.tokenizer.encode(word)[1:-1]
                    sentence += tokens
                    tags += [self.label2id[sent_label]] + DUMMY*(len(tokens)-1)
                    len_seq += 1

                else:
                    # ignore outliers (too long sequences)
                    # although their words were included in vocab... remove?
                    if len_seq > self.max_len:
                        pass
                    else:
                        sentence = CLS + sentence + SEP
                        tags = DUMMY + tags + DUMMY
                        self.id2seq[len(self.id2seq)] = tuple(sentence)
                        data[len(self.id2seq)-1] = tuple(tags)
                        
                    sentence = []
                    tags = []
                    len_seq = 0
                    
        return data 
 
    

class Results():
    """
    Object that stores all partial inputs and whose methods estimate the 
    evaluation metrics of incrementality.
    """
    def __init__(self, loader, model, my_device, label_pad_id, corpus,
                                  seq2seq, prophecies=None):
        print('Getting incremental results...')
        
        self.name = model._get_name() + '_' + corpus.task
        self.seq2seq = seq2seq
        # metrics in Baumann & Schlangen, 2011
        if not prophecies:
            if hasattr(model, 'bert_model'):
                self.results = get_bert_partial_outputs(loader, model, my_device, 
                                               label_pad_id, corpus.word2id['<pad>'],
                                               seq2seq)
            else:    
                # outputs using partial, incremental inputs
                self.results = get_partial_outputs(loader, model, my_device, 
                                               label_pad_id, seq2seq)
        else:
            self.name += 'withProphecies'
            if hasattr(model, 'bert_model'):
                self.results = get_bert_partial_outputs_with_prophecies(prophecies, 
                                                               loader, model, 
                                                               my_device, 
                                                               corpus, seq2seq)
            else:
                # outputs using GPT2 prophecies
                self.results = get_partial_outputs_with_prophecies(prophecies, 
                                                                   loader, model, 
                                                                   my_device, 
                                                                   corpus, seq2seq)
        self.lens = {key:value.shape[0] for key, value in 
                     self.results['partial_outputs'].items()}
        
        self.estimate_edit_overheads()
        self.estimate_correction_times()
        self.estimate_edit_overhead_ratio()
        self.estimate_rate_of_reanalysis()
        self.estimate_relative_correctness()
        
        self.perc_accurate = len([x for x in 
                                  self.results['accuracy'].values() if x == 1]) / len(self.results['accuracy'])
    
    def stats(self, metric_dict, only_correct=False, only_incorrect=False):
        """
        Estimates mean and standard deviation of values in a dictionary.

        Parameters
        ----------
        metric_dict : dict
            Dictionary mapping from sequence index to a metric.
        only_correct : bool, optional
            Only consider cases in which final outputs are correct with respect
            to the gold labels. The default is False.
        only_incorrect : bool, optional
            Only consider cases in which final outputs are incorrect with respect
            to the gold labels. The default is False.

        Returns
        -------
        mean : TYPE
            DESCRIPTION.
        std : TYPE
            DESCRIPTION.

        """
        if only_correct:
            
            mean = np.mean([v for k, v in metric_dict.items()
                            if self.results['accuracy'][k]==1])
            
            std = np.std([v for k, v in metric_dict.items()
                            if self.results['accuracy'][k]==1])
            
        elif only_incorrect:
            mean = np.mean([v for k, v in metric_dict.items()
                            if self.results['accuracy'][k]!=1])
            std = np.std([v for k, v in metric_dict.items()
                            if self.results['accuracy'][k]!=1])
        else:
            mean = np.mean(list(metric_dict.values()))
            std = np.std(list(metric_dict.values()))
            
        return mean, std
    
    def estimate_relative_correctness(self):
        """
        Creates dictionaries of relative correctness (RC), delay of t={0,1,2}.
        RC is the proportion of partial inputs that are correct w.r.t. the 
        final, non-incremental output.

        Returns
        -------
        None.

        """
        self.relative_correctness = {}
        self.relative_correctness_d1 = {}
        self.relative_correctness_d2 = {}
        
        for idx, outputs in self.results['partial_outputs'].items():
            self.relative_correctness[idx] = self._get_relative_correctness(outputs)
            self.relative_correctness_d1[idx] = self._get_relative_correctness_d1(outputs)
            self.relative_correctness_d2[idx] = self._get_relative_correctness_d2(outputs)
            
    def _get_relative_correctness(self, outputs):
        
        correct_guesses = [np.array_equal(outputs[i][:i+1], outputs[-1][:i+1]) for i in range(outputs.shape[0])]
        r_correctness = np.mean(correct_guesses)
        
        return r_correctness
    
    def _get_relative_correctness_d1(self, outputs):
        
        if outputs.shape[0] == 1:
            return 1
        
        correct_guesses = [np.array_equal(outputs[i][:i], outputs[-1][:i]) for i in range(1,outputs.shape[0])]
        r_correctness = np.mean(correct_guesses)
        
        return r_correctness
    
    def _get_relative_correctness_d2(self, outputs):
        
        if outputs.shape[0] == 1:
            return 1
        if outputs.shape[0] == 2:
            return 1
        
        correct_guesses = [np.array_equal(outputs[i][:i-1], outputs[-1][:i-1]) for i in range(2,outputs.shape[0])]
        r_correctness = np.mean(correct_guesses)

        return r_correctness
    
    def estimate_rate_of_reanalysis(self):
        """
        Creates dictionaries of rate of reanalysis (RR), delay of t={0,1,2}.
        RR is the proportion of edits over all possible edits that could have 
        been made.

        Returns
        -------
        None.

        """
       
        self.rate_of_reanalysis = {}
        self.rate_of_reanalysis_d1 = {}
        self.rate_of_reanalysis_d2 = {}
        
        for idx, changes in self.results['log_changes'].items():
            self.rate_of_reanalysis[idx] = self._get_rate_of_reanalysis(changes)
            self.rate_of_reanalysis_d1[idx] = self._get_rate_of_reanalysis_d1(changes)
            self.rate_of_reanalysis_d2[idx] = self._get_rate_of_reanalysis_d2(changes)
        
    def _get_rate_of_reanalysis(self, changes):
        
        seq_len = changes.shape[0]
        if self.seq2seq:
            possible_changes = ((seq_len - 1) * (seq_len)) / 2
            number_of_changes = changes.sum() - seq_len # substitutions - necessary additions
        else:
            possible_changes = seq_len-1
            number_of_changes = changes.sum() - 1 # substitutions - necessary additions
        
        # sequence of length 1
        if possible_changes == 0:
            return 0
        return number_of_changes / possible_changes
    
    def _get_rate_of_reanalysis_d1(self, changes):
        
        seq_len = changes.shape[0]
        if self.seq2seq:
            possible_changes = ((seq_len - 1) * (seq_len)) / 2
            number_of_changes = (changes.sum() 
                                 - changes.diagonal(-1).sum()
                                 - seq_len) # substitutions - necessary additions
        else:
            possible_changes = seq_len-1
            number_of_changes = (changes.sum() 
                                 - changes.diagonal(-1).sum() 
                                 - 1) # substitutions - necessary additions
        
        # sequence of length 1
        if possible_changes == 0:
            return 0
        return number_of_changes / possible_changes
    
    def _get_rate_of_reanalysis_d2(self, changes):
        
        seq_len = changes.shape[0]
        if self.seq2seq:
            possible_changes = ((seq_len - 1) * (seq_len)) / 2
            number_of_changes = (changes.sum() 
                                 - changes.diagonal(-1).sum() 
                                 - changes.diagonal(-2).sum()
                                 - seq_len) # substitutions - necessary additions
        else:
            possible_changes = seq_len-1
            number_of_changes = (changes.sum()
                                 - changes.diagonal(-1).sum()
                                 - changes.diagonal(-2).sum()
                                 - 1) # substitutions - necessary additions
        
        # sequence of length 1
        if possible_changes == 0:
            return 0
        return number_of_changes / possible_changes
    

    
    def estimate_edit_overhead_ratio(self):
        """
        Creates dictionaries of edit overhead ratio (EOR), delay of t={0,1,2}.
        EOR is the number of unnecessary edits (substitutions) divided by
        the number of necessary edits (additions).

        Returns
        -------
        None.

        """
       
        self.edit_overhead_ratio = {}
        self.edit_overhead_ratio_d1 = {}
        self.edit_overhead_ratio_d2 = {}

        for idx, changes in self.results['log_changes'].items():
            self.edit_overhead_ratio[idx] = self._get_edit_overhead_ratio(changes)
            self.edit_overhead_ratio_d1[idx] = self._get_edit_overhead_ratio_d1(changes)
            self.edit_overhead_ratio_d2[idx] = self._get_edit_overhead_ratio_d2(changes)
        
    def _get_edit_overhead_ratio(self, changes):
        
        necessary_additions = changes.diagonal().sum()
        unnecessary_substitutions = changes.sum() - changes.diagonal().sum()
        
        return unnecessary_substitutions / necessary_additions
    
    def _get_edit_overhead_ratio_d1(self, changes):
        
        necessary_additions = changes.diagonal().sum()
        unnecessary_substitutions = (changes.sum() - changes.diagonal().sum() 
                                     - changes.diagonal(-1).sum())
        
        return unnecessary_substitutions / necessary_additions
    
    def _get_edit_overhead_ratio_d2(self, changes):
        
        necessary_additions = changes.diagonal().sum()
        unnecessary_substitutions = (changes.sum() - changes.diagonal().sum() 
                                     - changes.diagonal(-1).sum()
                                     - changes.diagonal(-2).sum())
        
        return unnecessary_substitutions / necessary_additions
    
        
    def estimate_edit_overheads(self):
        """
        Creates dictionaries of edit overhead (EO), delay of t={0,1,2}.
        EO is the number of unnecessary edits (substitutions) divided by
        the total number edits (additions + substitutions).

        Returns
        -------
        None.

        """
        
        self.edit_overhead = {}
        self.edit_overhead_d1 = {}
        self.edit_overhead_d2 = {}

        for idx, changes in self.results['log_changes'].items():
            self.edit_overhead[idx] = self._get_edit_overhead(changes)
            self.edit_overhead_d1[idx] = self._get_edit_overhead_d1(changes)
            self.edit_overhead_d2[idx] = self._get_edit_overhead_d2(changes)
    
    # TODO merge numa funcao só com delta como parâmetro
    def _get_edit_overhead(self, changes):
        return (changes.sum() - changes.diagonal().sum())/changes.sum()
    
    def _get_edit_overhead_d1(self, changes):
        return (changes.sum() - changes.diagonal().sum() 
                - changes.diagonal(-1).sum())/changes.sum()
    
    def _get_edit_overhead_d2(self, changes):
        return (changes.sum() - changes.diagonal().sum() 
                - changes.diagonal(-1).sum() 
                - changes.diagonal(-2).sum())/changes.sum() 
    
    
    def mean_edit_overhead_perlength(self, eo_dict, only_correct=False, only_incorrect=False):
        
        len2eo = {i:[] for i in range(1, max(self.lens.values())+1)}
        
        if only_correct:
            for idx, eo in eo_dict.items():
                if self.results['accuracy'][idx] == 1:
                    len2eo[self.lens[idx]].append(eo)
        elif only_incorrect:
            for idx, eo in eo_dict.items():
                if self.results['accuracy'][idx] != 1:
                    len2eo[self.lens[idx]].append(eo)
        else:
            for idx, eo in eo_dict.items():
                len2eo[self.lens[idx]].append(eo)
        # length is position in this list        
        mean_eos = [np.nan]+ [np.mean(len2eo[i]) for i in range(1, max(self.lens.values())+1)]
        
        return mean_eos
 
    
    def estimate_correction_times(self):
        """
        Creates dictionaries of correction time score (CT), delay of t={0,1,2}.
        CTScore is a score of the sum of the number of steps it took for a 
        final decision to be reached for each label, divided by the number of
        all possible steps.

        Returns
        -------
        None.

        """
        self.correction_time_pertimestep = {}
        self.correction_time_score = {}
        
        for idx, outputs in self.results['partial_outputs'].items():
            ct = self._get_correction_times(outputs)
            ct_len = len(ct)
            self.correction_time_pertimestep[idx] = ct
            # score I invented to account for correction time in different time
            # steps and different sequence lengths. 0 means all gueses were 
            # right from the beginning, 1 means guesses were only correct in 
            # last output. Lower is better (it takes less time to make correct
            # decisions)
            if self.seq2seq:
                if ct_len==1: # sentences with len 1 will have score 0
                   self.correction_time_score[idx] = 0
                else:
                   self.correction_time_score[idx] = np.sum(ct) / (((ct_len-1)*ct_len)/2) # (len-1)+(len-1)+...+1, total of highest correction times for all time steps
            else:
                # if seq2label, simply divide by possible number of corrections
                if outputs.shape[0]==1:
                    self.correction_time_score[idx] = ct[0]
                else:
                    self.correction_time_score[idx] = ct[0] / (outputs.shape[0]-1)
                
    def _get_correction_times(self, outputs):
        
        len_seq = outputs.shape[0]
        FD = [] # final decision, F0 is always position of that input
        for c, column in enumerate(outputs.T):
            # final seq, correct input was chosen and did not change anymore
            last_group=[tuple(g) for _, g in itertools.groupby(column)][-1]
            # time step (counting from 0) when final decision was made
            # meaning how many steps were necessary to get to correct label
            # 0 means no change happened
            FD.append((len_seq - c) - len(last_group))
            
        return FD
    
    
    def mean_correction_time_perlength(self, ct_dict, only_correct=False, only_incorrect=False):
        
        len2ct = {i:[] for i in range(1, max(self.lens.values())+1)}
        
        if only_correct:
            for idx, ct in ct_dict.items():
                if self.results['accuracy'][idx] == 1:
                    len2ct[self.lens[idx]].append(ct)
        elif only_incorrect:
            for idx, ct in ct_dict.items():
                if self.results['accuracy'][idx] != 1:
                    len2ct[self.lens[idx]].append(ct)
        else:
            for idx, ct in ct_dict.items():
                len2ct[self.lens[idx]].append(ct)
        # length is position in this list        
        mean_cts = [np.nan]+ [np.mean(len2ct[i]) for i in range(1, max(self.lens.values())+1)]

        return mean_cts
 
    
    def print_metrics(self, name, experiment=None):
        
        mean_eo, std_eo = self.stats(self.edit_overhead)
        mean_eo_d1, std_eo_d1 = self.stats(self.edit_overhead_d1)
        mean_eo_d2, std_eo_d2 = self.stats(self.edit_overhead_d2)
        mean_eor, std_eor = self.stats(self.edit_overhead_ratio)
        mean_eor_d1, std_eor_d1 = self.stats(self.edit_overhead_ratio_d1)
        mean_eor_d2, std_eor_d2 = self.stats(self.edit_overhead_ratio_d2)
        mean_rr, std_rr = self.stats(self.rate_of_reanalysis)
        mean_rr_d1, std_rr_d1 = self.stats(self.rate_of_reanalysis_d1)
        mean_rr_d2, std_rr_d2 = self.stats(self.rate_of_reanalysis_d2)
        mean_rc, std_rc = self.stats(self.relative_correctness)
        mean_rc_d1, std_rc_d1 = self.stats(self.relative_correctness_d1)
        mean_rc_d2, std_rc_d2 = self.stats(self.relative_correctness_d2)
        mean_ct, std_ct = self.stats(self.correction_time_score)
        
        mean_eo_c, std_eo_c = self.stats(self.edit_overhead, only_correct=True)
        mean_eo_d1_c, std_eo_d1_c = self.stats(self.edit_overhead_d1, only_correct=True)
        mean_eo_d2_c, std_eo_d2_c = self.stats(self.edit_overhead_d2, only_correct=True)
        mean_eor_c, std_eor_c = self.stats(self.edit_overhead_ratio, only_correct=True)
        mean_eor_d1_c, std_eor_d1_c = self.stats(self.edit_overhead_ratio_d1, only_correct=True)
        mean_eor_d2_c, std_eor_d2_c = self.stats(self.edit_overhead_ratio_d2, only_correct=True)
        mean_rr_c, std_rr_c =  self.stats(self.rate_of_reanalysis, only_correct=True)
        mean_rr_d1_c, std_rr_d1_c =  self.stats(self.rate_of_reanalysis_d1, only_correct=True)
        mean_rr_d2_c, std_rr_d2_c =  self.stats(self.rate_of_reanalysis_d2, only_correct=True)
        mean_rc_c, std_rc_c = self.stats(self.relative_correctness, only_correct=True)
        mean_rc_d1_c, std_rc_d1_c = self.stats(self.relative_correctness_d1, only_correct=True)
        mean_rc_d2_c, std_rc_d2_c = self.stats(self.relative_correctness_d2, only_correct=True)
        mean_ct_c, std_ct_c = self.stats(self.correction_time_score, only_correct=True)
        
        mean_eo_i, std_eo_i = self.stats(self.edit_overhead, only_incorrect=True)
        mean_eo_d1_i, std_eo_d1_i = self.stats(self.edit_overhead_d1, only_incorrect=True)
        mean_eo_d2_i, std_eo_d2_i = self.stats(self.edit_overhead_d2, only_incorrect=True)
        mean_eor_i, std_eor_i = self.stats(self.edit_overhead_ratio, only_incorrect=True)
        mean_eor_d1_i, std_eor_d1_i = self.stats(self.edit_overhead_ratio_d1, only_incorrect=True)
        mean_eor_d2_i, std_eor_d2_i = self.stats(self.edit_overhead_ratio_d2, only_incorrect=True)
        mean_rr_i, std_rr_i = self.stats(self.rate_of_reanalysis, only_incorrect=True)
        mean_rr_d1_i, std_rr_d1_i = self.stats(self.rate_of_reanalysis_d1, only_incorrect=True)
        mean_rr_d2_i, std_rr_d2_i = self.stats(self.rate_of_reanalysis_d2, only_incorrect=True)
        mean_rc_i, std_rc_i = self.stats(self.relative_correctness, only_incorrect=True)
        mean_rc_d1_i, std_rc_d1_i = self.stats(self.relative_correctness_d1, only_incorrect=True)
        mean_rc_d2_i, std_rc_d2_i = self.stats(self.relative_correctness_d2, only_incorrect=True)
        mean_ct_i, std_ct_i = self.stats(self.correction_time_score, only_incorrect=True)
        
        with open('incrementality_metrics/'+name+'.txt', 'w') as file:
            file.write('All outputs \n')
            file.write('Metric \t Mean \t STD \n')
            file.write('EO \t {}\t{} \n'.format(mean_eo, std_eo))
            file.write('EOD1 \t {}\t{} \n'.format(mean_eo_d1, std_eo_d1))
            file.write('EOD2 \t {}\t{} \n'.format(mean_eo_d2, std_eo_d2))
            file.write('EOR \t {}\t{} \n'.format(mean_eor, std_eor))
            file.write('EORD1 \t {}\t{} \n'.format(mean_eor_d1, std_eor_d1))
            file.write('EORD2 \t {}\t{} \n'.format(mean_eor_d2, std_eor_d2))
            file.write('RR \t {}\t{} \n'.format(mean_rr, std_rr))
            file.write('RRD1 \t {}\t{} \n'.format(mean_rr_d1, std_rr_d1))
            file.write('RRD2 \t {}\t{} \n'.format(mean_rr_d2, std_rr_d2))
            file.write('RC \t {}\t{} \n'.format(mean_rc, std_rc))
            file.write('RCD1 \t {}\t{} \n'.format(mean_rc_d1, std_rc_d1))
            file.write('RCD2 \t {}\t{} \n'.format(mean_rc_d2, std_rc_d2))
            file.write('CTscore \t {}\t{} \n\n'.format(mean_ct, std_ct))
            
            file.write('Correct outputs \n')
            file.write('Metric \t Mean \t STD \n')
            file.write('EO \t {}\t{} \n'.format(mean_eo_c, std_eo_c))
            file.write('EOD1 \t {}\t{} \n'.format(mean_eo_d1_c, std_eo_d1_c))
            file.write('EOD2 \t {}\t{} \n'.format(mean_eo_d2_c, std_eo_d2_c))
            file.write('EOR \t {}\t{} \n'.format(mean_eor_c, std_eor_c))
            file.write('EORD1 \t {}\t{} \n'.format(mean_eor_d1_c, std_eor_d1_c))
            file.write('EORD2 \t {}\t{} \n'.format(mean_eor_d2_c, std_eor_d2_c))
            file.write('RR \t {}\t{} \n'.format(mean_rr_c, std_rr_c))
            file.write('RRD1 \t {}\t{} \n'.format(mean_rr_d1_c, std_rr_d1_c))
            file.write('RRD1 \t {}\t{} \n'.format(mean_rr_d2_c, std_rr_d2_c))
            file.write('RC \t {}\t{} \n'.format(mean_rc_c, std_rc_c))
            file.write('RCD1 \t {}\t{} \n'.format(mean_rc_d1_c, std_rc_d1_c))
            file.write('RCD2 \t {}\t{} \n'.format(mean_rc_d2_c, std_rc_d2_c))
            file.write('CTscore \t {}\t{} \n\n'.format(mean_ct_c, std_ct_c))
            
            file.write('Incorrect outputs \n')
            file.write('Metric \t Mean \t STD \n')
            file.write('EO \t {}\t{} \n'.format(mean_eo_i, std_eo_i))
            file.write('EOD1 \t {}\t{} \n'.format(mean_eo_d1_i, std_eo_d1_i))
            file.write('EOD2 \t {}\t{} \n'.format(mean_eo_d2_i, std_eo_d2_i))
            file.write('EOR \t {}\t{} \n'.format(mean_eor_i, std_eor_i))
            file.write('EOR_d1 \t {}\t{} \n'.format(mean_eor_d1_i, std_eor_d1_i))
            file.write('EOR_d2 \t {}\t{} \n'.format(mean_eor_d2_i, std_eor_d2_i))
            file.write('RR \t {}\t{} \n'.format(mean_rr_i, std_rr_i))
            file.write('RRD1 \t {}\t{} \n'.format(mean_rr_d1_i, std_rr_d1_i))
            file.write('RRD2 \t {}\t{} \n'.format(mean_rr_d2_i, std_rr_d2_i))
            file.write('RC \t {}\t{} \n\n'.format(mean_rc_i, std_rc_i))
            file.write('RCD1 \t {}\t{} \n\n'.format(mean_rc_i, std_rc_i))
            file.write('RCD2 \t {}\t{} \n\n'.format(mean_rc_i, std_rc_i))
            file.write('CTscore \t {}\t{} \n'.format(mean_ct_i, std_ct_i))
        
        mode = ''
        if 'gpt' in name:
            mode = '_gpt'
            
        if experiment:
            experiment.log_metric("mean_EO"+mode, mean_eo)
            experiment.log_metric("mean_EO_d1"+mode, mean_eo_d1)
            experiment.log_metric("mean_EO_d2"+mode, mean_eo_d2)
            experiment.log_metric("mean_EOR"+mode, mean_eor)
            experiment.log_metric("mean_EOR_d1"+mode, mean_eor_d1)
            experiment.log_metric("mean_EOR_d2"+mode, mean_eor_d2)
            experiment.log_metric("mean_RR"+mode, mean_rr)
            experiment.log_metric("mean_RR_d1"+mode, mean_rr_d1)
            experiment.log_metric("mean_RR_d2"+mode, mean_rr_d2)
            experiment.log_metric("mean_RC"+mode, mean_rc)
            experiment.log_metric("mean_RC_d1"+mode, mean_rc_d1)
            experiment.log_metric("mean_RC_d2"+mode, mean_rc_d2)
            experiment.log_metric("mean_CT"+mode, mean_ct)
            
            experiment.log_metric("mean_EO_c"+mode, mean_eo_c)
            experiment.log_metric("mean_EO_d1_c"+mode, mean_eo_d1_c)
            experiment.log_metric("mean_EO_d2_c"+mode, mean_eo_d2_c)
            experiment.log_metric("mean_EOR_c"+mode, mean_eor_c)
            experiment.log_metric("mean_EOR_d1_c"+mode, mean_eor_d1_c)
            experiment.log_metric("mean_EOR_d2_c"+mode, mean_eor_d2_c)
            experiment.log_metric("mean_RR_c"+mode, mean_rr_c)
            experiment.log_metric("mean_RR_d1_c"+mode, mean_rr_d1_c)
            experiment.log_metric("mean_RR_d2_c"+mode, mean_rr_d2_c)
            experiment.log_metric("mean_RC_c"+mode, mean_rc_c)
            experiment.log_metric("mean_RC_d1_c"+mode, mean_rc_d1_c)
            experiment.log_metric("mean_RC_d2_c"+mode, mean_rc_d2_c)
            experiment.log_metric("mean_CT_c"+mode, mean_ct_c)
            
            experiment.log_metric("mean_EO_i"+mode, std_eo_i)
            experiment.log_metric("mean_EO_d1_i"+mode, mean_eo_d1_i)
            experiment.log_metric("mean_EO_d2_i"+mode, mean_eo_d2_i)
            experiment.log_metric("mean_EOR_i"+mode, mean_eor_i)
            experiment.log_metric("mean_EOR_d1_i"+mode, mean_eor_d1_i)
            experiment.log_metric("mean_EOR_d2_i"+mode, mean_eor_d2_i)
            experiment.log_metric("mean_RR_i"+mode, mean_rr_i)
            experiment.log_metric("mean_RR_d1_i"+mode, mean_rr_d1_i)
            experiment.log_metric("mean_RR_d2_i"+mode, mean_rr_d2_i)
            experiment.log_metric("mean_RC_i"+mode, mean_rc_i)
            experiment.log_metric("mean_RC_d1_i"+mode, mean_rc_d1_i)
            experiment.log_metric("mean_RC_d2_i"+mode, mean_rc_d2_i)
            experiment.log_metric("mean_CT_i"+mode, mean_ct_i)
            
            
            experiment.log_metric("correct_outputs"+mode, self.perc_accurate)
            
             
        if experiment:
            experiment.log_asset('incrementality_metrics/'+name+'.txt')
            