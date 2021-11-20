from comet_ml import Experiment

import torch
import torch.optim as optim
import numpy as np
import pickle
import argparse
import time

from structs import Corpus, Results, Corpus4Bert
from models import vanillaLSTM, LSTMCRF, BERT
from utils import load_data, train, test, evaluate_nn, voidExperiment


############################### SETTINGS #####################################

# Define general parameters
parser = argparse.ArgumentParser(description=
                                 'Evaluation of incremental outputs in NNs')
# task
parser.add_argument('--task', type=str, default='chunk',
                    help='type of task: snips_slot snips_intent, atis_slot, \
                        atis_intent, chunk, proscons, srl_nw_wsj, \
                        pos_nw_wsj, ner_nw_wsj, sent_negpos')
parser.add_argument('--only_training', action='store_true', 
                    help='train model only, no incrementality evaluation')
parser.add_argument('--comet_track', action='store_true', 
                    help='log data to comet.ml')
parser.add_argument('--truncated_training', action='store_true', 
                    help='sample truncated inputs during training')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='choose a specific device')
parser.add_argument('--outliers', type=int, default=200,
                    help='len above which sentences are ignored')

# nn model
# in case of choosing BERT, most of these hyperparameters are ignored
# since it has its own structure
parser.add_argument('--model', type=str, default='vanilla_bilstm',
                    help='type of LSTM: vanilla_lstm, vanilla_bilstm, \
                        lstm_crf, bilstm_crf, bert-base-{cased, uncased}')
parser.add_argument('--dim_emb', type=int, default=300,
                    help='dimenstion of word embeddings')
parser.add_argument('--dim_hid', type=int, default=200,
                    help='dimension of hidden layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of lstm layers')
parser.add_argument('--epochs', type=int, default=50,
                    help='training iterations over dataset') 
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='size of gradient for clipping, 0 for no clipping')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout probability')
parser.add_argument('--no_glove', action='store_true',
                    help='do not use GloVe embeddings')
parser.add_argument('--freeze', action='store_true',
                    help='do not update GloVe embeddings during training')

args = parser.parse_args()

# seq2label outputs one label per sequence, instead of one per word
seq2label_tasks = ['atis_intent', 'snips_intent', 'proscons', 'sent_negpos']
# tasks that use accuracy instead of F1 for evaluation during training
eval_with_acc = ['atis_intent', 'snips_intent', 'pos_nw_wsj', 'proscons',
                 'sent_negpos']

if 'bert' in args.model:
    assert args.dropout == 0.1 # value in original model
    
# comet.ml experiment information
if args.comet_track:
    experiment = Experiment(api_key="",
                        project_name="", 
                        workspace="",
                        auto_output_logging="simple",
                        log_git_metadata=False,
                        log_git_patch=False)
    if experiment.alive is False:
        raise Exception("Could not connect to comet.ml!")
    # disable automatic logging
    # experiment.auto_param_logging=False
    # parameters all args above as parameters
    experiment.log_parameters(args.__dict__)
    experiment.add_tags([args.task, args.model])   
else:
    # pseudoclass to enable "with experiment.train()" when not using comet
    experiment = voidExperiment() 

# GloVe has 4 valid embedding dimensions, assert that one of them is chosen
if not args.no_glove:
    assert args.dim_emb in [50, 100, 200, 300], 'Choose valid GloVe dimension'

print('\nEvaluating incremental outputs of task: {}, with model {}.\n'.format(
                                                        args.task, args.model))

seq2seq=True
if args.task in seq2label_tasks:
    assert args.model not in ['lstm_crf', 'bilstm_crf'], 'CRF cannot be used with seq2label task'
    seq2seq = False 

# use GPU if it is available
#my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_device = torch.device(args.device)

print('Using', my_device)
# set manual seed for reproducibility
torch.manual_seed(2204)

if 'bert' in args.model:
    # read corpus with no UNK tokens, neither in training nor in valid/test
    corpus = Corpus4Bert(args.model, args.task, seq2seq, max_len=args.outliers)  
else:    
    # read corpus and create dataloaders
    corpus = Corpus(args.task, seq2seq, max_len=args.outliers)
    
train_loader, valid_loader, test_loader = load_data(corpus, 
                                                    batch_size=args.batch_size,
                                                    sample=args.truncated_training,
                                                    model=args.model)
size_vocab = len(corpus.word2id) # will be 1 in bert, not a real count because we use their tokenizer and pretrained vocab size
# beware that labels2id has an extra symbol for padding that is not a tag,
# so we subtract 1 from number of labels
n_labels = len(corpus.label2id)-1 
pad_id = corpus.word2id['<pad>']
label_pad_id = corpus.label2id['<pad>']

if args.comet_track:
    experiment.log_other("size_vocab", size_vocab)
    experiment.log_other("n_labels", n_labels)
    experiment.log_other("size_trainset", len(corpus.train))
    experiment.log_other("size_validset", len(corpus.valid))
    experiment.log_other("size_testset", len(corpus.test))

# Create NN model and send it to my_device
print('Building model...')
if args.model == 'vanilla_lstm':
    model = vanillaLSTM(size_vocab, args.dim_emb, args.dim_hid, args.nlayers, 
                           n_labels, args.dropout, pad_id, corpus,
                           no_glove=args.no_glove, freeze=args.freeze,
                           bidirectional=False).to(my_device)
elif args.model == 'vanilla_bilstm':
    model = vanillaLSTM(size_vocab, args.dim_emb, args.dim_hid, args.nlayers, 
                           n_labels, args.dropout, pad_id, corpus, 
                           no_glove=args.no_glove, freeze=args.freeze,
                           bidirectional=True).to(my_device)
elif args.model == 'lstm_crf':
    model = LSTMCRF(size_vocab, args.dim_emb, args.dim_hid, args.nlayers, 
                           n_labels, args.dropout, pad_id, corpus, 
                           no_glove=args.no_glove, freeze=args.freeze,
                           bidirectional=False).to(my_device)
elif args.model == 'bilstm_crf':
    model = LSTMCRF(size_vocab, args.dim_emb, args.dim_hid, args.nlayers, 
                           n_labels, args.dropout, pad_id, corpus, 
                           no_glove=args.no_glove, freeze=args.freeze,
                           bidirectional=True).to(my_device)  
elif 'bert' in args.model:
    model = BERT(n_labels, corpus, seq2seq, args.model, args.dropout).to(my_device)
    # freeze bert's pretrained parameters
    #for param in model.encoder.bert.parameters():
    #    param.requires_grad=False
else:
    raise 'Choose a model among the five options.'

# choose minimization method
#if 'bert' in args.model:
    # update only classifier model and use fixed pretrained models
    # https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
#    param_optimizer = list(model.encoder.classifier.named_parameters()) 
#    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
#    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr)
#else:
optimizer = optim.Adam(model.parameters(), lr=args.lr)      

################################ LEARNING ####################################

print('Learning started.')
start_learn = time.time()

model_name = args.task + '_' + args.model
if args.truncated_training == True:
    model_name += '_truncated-training'
# reverse dict for seqeval
id2label = {id:label for label, id in corpus.label2id.items()}

# if valid performance does not improve for x iterations, stop training
EARLY_STOPPING = 10
if args.comet_track:
    experiment.log_other("early_stopping", EARLY_STOPPING)

valid_performance = []
best_valid_performance = -np.inf

# begin training
for epoch in range(args.epochs):
    start_epoch = time.time()
    print('Epoch {}'.format(epoch+1))
    
    # training
    with experiment.train():
        train_loss, preds, golds = train(train_loader, model, my_device, 
                                                 optimizer, args.clip, 
                                                 label_pad_id, seq2seq)
        train_acc, train_f1 = evaluate_nn(golds, preds, id2label)
        if args.comet_track:
            experiment.log_metric("nl_loss", train_loss, step=epoch)
            experiment.log_metric("acc", train_acc, step=epoch)
            experiment.log_metric("f1", train_f1, step=epoch)
        
    # validation
    with experiment.validate():
        valid_loss, preds, golds = test(valid_loader, model, my_device, 
                                                label_pad_id, seq2seq)
        valid_acc, valid_f1 = evaluate_nn(golds, preds, id2label)
        # metrics for early stopping depending on task
        if args.task in eval_with_acc:
            valid_performance.append(valid_acc)
        else:
            valid_performance.append(valid_f1)
        if args.comet_track:
            experiment.log_metric("nl_loss", valid_loss, step=epoch)
            experiment.log_metric("acc", valid_acc, step=epoch)
            experiment.log_metric("f1", valid_f1, step=epoch)
    
    # store model with best performance on valid
    if valid_performance[-1] > best_valid_performance:
        best_valid_performance = valid_performance[-1]
        torch.save(model.state_dict(), 'models/'+model_name+'_bestmodel.pt')
        
    # early stopping, if performance did not improve for a while, stop training
    if np.argmax(valid_performance) < len(valid_performance) - EARLY_STOPPING:
        break
    
    elapsed_epoch = time.time() - start_epoch
    print(' {0[0]:.0f} min {0[1]:.0f} secs. \n'.format(
                                                    divmod(elapsed_epoch, 60)))
if args.comet_track:
            experiment.log_metric("best_valid_performance", 
                                  best_valid_performance)
            
print('Stopped at epoch {}.\n'.format(epoch+1))    
if args.comet_track:
    experiment.log_metric("last_epoch", epoch)    
elapsed_learn = time.time() - start_learn
print('Learning took {0[0]:.0f} min {0[1]:.0f} secs. \n'.format(
                                                    divmod(elapsed_learn, 60)))

# load best model during training
model.load_state_dict(torch.load('models/'+model_name+'_bestmodel.pt'))
model.eval()
if args.comet_track:
    experiment.log_model('best_model', 'models/'+model_name+'_bestmodel.pt')

# check how well it does on test set
with experiment.test():
    test_loss, preds, golds = test(test_loader, model, my_device, label_pad_id, 
                                    seq2seq, dataset='test   ')
    test_acc, test_f1 = evaluate_nn(golds, preds, id2label)
    if args.comet_track:
        experiment.log_metric("nl_loss", test_loss)
        experiment.log_metric("acc", test_acc)
        experiment.log_metric("f1", test_f1)

############################ EVALUATING INCREMENTALITY #######################

if not args.only_training:
    print('Incremental processing evaluation started.')
    
    if not args.comet_track:
        experiment = None
    
    # outputs using partial, incremental inputs
    partial_outputs = Results(test_loader, model, my_device, label_pad_id,
                                     corpus, seq2seq, prophecies=None)
    partial_outputs.print_metrics(model_name, experiment)
    
    pickle.dump(partial_outputs, 
                open('outputs/results_'+model_name, 'wb'))
    if args.comet_track:
        experiment.log_asset('outputs/results_'+model_name, 
                             'results_partialInputs_'+model_name)
    
    # outputs using GPT2 prophecies
    prophecies = pickle.load(open('prophecies/gpt2Prophecies_'+args.task+'_testset-withOutliers', 'rb'))
    prophecies_outputs = Results(test_loader, model, 
                                            my_device, label_pad_id, corpus, 
                                            seq2seq, prophecies) 
    prophecies_outputs.print_metrics(model_name+'_gpt', experiment)
    
    pickle.dump(prophecies_outputs, 
                open('outputs/resultsGPT_'+model_name, 'wb'))
    if args.comet_track:
        experiment.log_asset('outputs/resultsGPT_'+model_name,
                             "results_prophecies_"+model_name)

print('Finished!')
if args.comet_track:
    experiment.end()
# Sources:
# https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
# https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
