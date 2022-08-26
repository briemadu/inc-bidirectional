# WARNING

[May 5, 2022] The Edit Overhead metric with delay was being incorrectly computed in the code. It has been fixed in the code. A revision will come soon, but the conclusions remain unchanged.

# README #

Code used for running experiments in the research paper:

MADUREIRA, Brielen & SCHLANGEN, David. Incremental Processing in the Age of Non-Incremental Encoders: An Empirical Assessment of Bidirectional Models for Incremental NLU. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing.

Check the pdf in this repository.

### What is this repository for? ###

We can use this code to train a neural encoder and use it either for sequence tagging or sequence classification with 10 task/dataset combinations. After training, it retrieves the incrementality evaluation metrics in the test set. Two special configurations may be added: truncated training (we sample a length to use only a prefix of each sentence during training) and prophecies (we use GPT-2 language model to generate 'prophecies' for each prefix of the sequence, and feed this hypothetical continuation to the encoder during the estimation of incrementality metrics).


### How do I get set up? ###

## Set up

Specific Python installations:

* [PyTorch](https://pytorch.org/) (v. 1.3.1)
* [pytorch-crf](https://pypi.org/project/pytorch-crf/) (v. 0.7.2) to add a CRF layer on top of our neural network
* [seqeval](https://pypi.org/project/seqeval/) (v. 0.0.12) to estimate F1 score in BIO labeling scheme
* [transformers](https://github.com/huggingface/transformers) (v. 2.5.1) to generate prophecies and use BERT model
* [nltk](https://www.nltk.org/) (v. 3.4.5) for tokenization of generated prophecies
* [comet_ml](https://www.comet.ml/docs/quick-start/) if you want to log information of your experiment (optional)

If you want to use comet_ml, first include your api_key, project name and workspace on line 76 in main.py.

Download GloVe pre-trained embeddings [here](http://nlp.stanford.edu/data/glove.6B.zip) and unzip it in the main folder (replacing the currently empty glove.6B folder here).


# Download data

* [CoNLL 2000](https://www.clips.uantwerpen.be/conll2000/chunking/)
* [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
* [ATIS](https://www.aclweb.org/anthology/H90-1021.pdf)
* [SNIPS](https://github.com/sonos/nlu-benchmark)
* [Pros/Cons](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#datasets)
* [Positive/Negative](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)

For ATIS and SNIPS, we use the preprocessed data made available by [E et al. (2019)](https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU).

Preprocessing is needed to put them in the following format:

* Sequence tagging:
> token \t label \n token \t label \n
with an extra \n between sequences.

* Sequence classification:
> <LABEL>: atis_airfare \n token \n token \n
with an extra \n between sequences.

See examples in the data directory.

# Prepare data

As it is, the code can run experiments on:

* Sequence tagging
    * chunk, CoNLL 2000 (chunk)
    * named entity recognition, OntoNotes5.0, WSJ part (ner_nw_wsj)
    * pos tagging, OntoNotes5.0, WSJ part (pos_nw_wsj)
    * semantic role labeling, OntoNotes5.0, WSJ part (srl_nw_wsj)
    * slot filling, ATIS or SNIPS (atis_slot, snips_slot)
* Sequence classification
    * Sentiment (proscons or sent_negpos)
    * intent , ATIS or SNIPS (atis_intent, snips_intent)

If you have a new file, include the name in lines 65-69 in main.py, to specify whether it is a tagging or classification task and which evaluation function should the system use during training.

Data has to be split into three files (data/train/train.<task>, data/valid/valid.<task> and data/test/test.<task>), where <task> is one of the names in parenthesis above. All of them must follow the format above.

## How to run tests

> python3.py main.py

Use `--help` to check all possible arguments.
>  --task TASK           type of task: snips_slot snips_intent, atis_slot, atis_intent, chunk, proscons, srl_nw_wsj, pos_nw_wsj, ner_nw_wsj, sent_negpos

>  --only_training       train model only, no incrementality evaluation

>  --comet_track         log data to comet.ml

>  --truncated_training  sample truncated inputs during training

>  --device DEVICE       choose a specific device

>  --outliers OUTLIERS   len above which sentences are ignored

>  --model MODEL         type of LSTM: vanilla_lstm, vanilla_bilstm, lstm_crf, bilstm_crf, bert-base-{cased, uncased}

>  --dim_emb DIM_EMB     dimenstion of word embeddings

>  --dim_hid DIM_HID     dimension of hidden layer

>  --nlayers NLAYERS     number of lstm layers

>  --epochs EPOCHS       training iterations over dataset

>  --batch_size BATCH_SIZE batch size

>  --lr LR               initial learning rate

>  --clip CLIP           size of gradient for clipping, 0 for no clipping

>  --dropout DROPOUT     dropout probability

>  --no_glove            do not use GloVe embeddings

>  --freeze              do not update GloVe embeddings during training

### Citing this paper ###

```
@inproceedings{madureira-schlangen-2020-incremental,
    title = "Incremental Processing in the Age of Non-Incremental Encoders: An Empirical Assessment of Bidirectional Models for Incremental {NLU}",
    author = "Madureira, Brielen  and
      Schlangen, David",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.26",
    doi = "10.18653/v1/2020.emnlp-main.26",
    pages = "357--374",
}   
```
