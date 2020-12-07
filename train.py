import os
import sys
import argparse
import time
import random
import utils
import pdb
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data import PaddedTensorDataset
from data import TextLoader
from model import LSTMClassifier

# python train.py --data_dir /Users/zhaohuilee/Desktop/RA/2020-fall/ASGA/data_test/train --test_dir /Users/zhaohuilee/Desktop/RA/2020-fall/ASGA/data_test/test --batch_size 4 --num_epochs 5

# python train.py --data_dir /Users/zhaohuilee/Desktop/RA/2020-fall/ASGA/sem_data/training/3way --test_dir /Users/zhaohuilee/Desktop/RA/2020-fall/ASGA/sem_data/test/3way --batch_size 64 --hidden_dim 32 --char_dim 100 --num_epochs 50 --learning_rate 0.001

# python train.py --data_dir ./data/semeval2013-Task7-2and3way/training/2way/beetle --test_dir ./data/semeval2013-Task7-2and3way/test/2way/beetle/test-unseen-answers --batch_size 64 --hidden_dim 64 --char_dim 50 --num_epochs 50

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/semeval2013-Task7-5way/beetle/train/Core/test',
                                            help='data_directory')
    parser.add_argument('--test_dir', type=str,
                        default='/Users/zhaohuilee/Desktop/RA/2020-fall/ASGA/data/semeval2013-Task7-5way/beetle/test-unseen-questions/Core/test',
                        help='data_directory')
    parser.add_argument('--hidden_dim', type=int, default=32,
                                            help='LSTM hidden dimensions')
    parser.add_argument('--batch_size', type=int, default=8,
                                            help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=20,
                                            help='maximum number of epochs')
    parser.add_argument('--char_dim', type=int, default=100,
                                            help='character embedding dimensions')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                                            help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                                            help='weight_decay rate')
    parser.add_argument('--seed', type=int, default=123,
                                            help='seed for random initialisation')
    args = parser.parse_args()
    train(args)

def loadEMbeddingMatrix(vocab, typeToLoad, embed_size = 100):
    EMBEDDING_FILE = './model/glove.6B.50d.txt'
    if typeToLoad == 'glove' or typeToLoad == 'fasttext':
        embedding_index = dict()
        f = open(EMBEDDING_FILE)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
        f.close()
        print('Load %s word vectors.' % len(embedding_index))
        embedding_matrix = np.random.normal(0, 1, (len(vocab), embed_size))
        embeddedCount = 0
        for word, i in vocab.items():  # 词
            embedding_vector = embedding_index.get(word)
            # print(word)

            if embedding_vector is not None:
                # print("++++")
                # print(embedding_vector)
                embedding_matrix[i] = embedding_vector
                embeddedCount += 1
        print('total_embedded:', embeddedCount, 'commen words')

        return embedding_matrix

def apply(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss


def train_model(model, optimizer, train, dev, x_to_ix, y_to_ix, batch_size, max_epochs):
    criterion = nn.NLLLoss(size_average=False)
    for epoch in range(max_epochs):
        print('Epoch:', epoch)
        y_true = list()
        y_pred = list()
        # print(train)
        # print(x_to_ix)
        # print(y_to_ix)
        total_loss = 0
        #print(train)
        for batch, targets, lengths, raw_data in utils.create_dataset(train, x_to_ix, y_to_ix, batch_size=batch_size):
            batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
            model.zero_grad()
            pred, loss = apply(model, criterion, batch, targets, lengths)
            loss.backward()
            optimizer.step()
            
            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
        val_loss, val_acc = evaluate_validation_set(model, dev, x_to_ix, y_to_ix, criterion)
        print("Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(total_loss.data.float()/len(train), acc,
                                                                                val_loss, val_acc))
    return model


def evaluate_validation_set(model, devset, x_to_ix, y_to_ix, criterion):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in utils.create_dataset(devset, x_to_ix, y_to_ix, batch_size=1):
        batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
        pred, loss = apply(model, criterion, batch, targets, lengths)
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
        total_loss += loss
    acc = accuracy_score(y_true, y_pred)
    return total_loss.data.float()/len(devset), acc


def evaluate_test_set(model, test, x_to_ix, y_to_ix):
    y_true = list()
    y_pred = list()
    #print(test)
    for batch, targets, lengths, raw_data in utils.create_dataset(test, x_to_ix, y_to_ix, batch_size=1):
        batch, targets, lengths = utils.sort_batch(batch, targets, lengths)

        pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())

    print(len(y_true), len(y_pred))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


def train(args):
    random.seed(args.seed)
    data_loader = TextLoader(args.data_dir, args.test_dir)

    train_data = data_loader.train_data
    dev_data = data_loader.dev_data
    test_data = data_loader.test_data

    char_vocab = data_loader.token2id
    tag_vocab = data_loader.tag2id
    char_vocab_size = len(char_vocab)

    print('Training samples:', len(train_data))
    print('Valid samples:', len(dev_data))
    print('Test samples:', len(test_data))

    # print(char_vocab)
    # print(tag_vocab)
    # print(train_data)
    #assert 1==0
    embedding_matrix = loadEMbeddingMatrix(char_vocab, "glove", args.char_dim)
    print("-------------")
    print(embedding_matrix.shape)
    model = LSTMClassifier(char_vocab_size, args.char_dim, args.hidden_dim, len(tag_vocab), embedding_matrix)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model = train_model(model, optimizer, train_data, dev_data, char_vocab, tag_vocab, args.batch_size, args.num_epochs)

    evaluate_test_set(model, test_data, char_vocab, tag_vocab)


if __name__ == '__main__':
    main()
