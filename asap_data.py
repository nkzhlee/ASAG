import os
import csv
import sys
import math
import random
import argparse
import operator
import pdb
import xml4h
import spacy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from collections import defaultdict
from collections import Counter
from torch.autograd import Variable

class TextLoader:
    def __init__(self, data_dir, test_dir):
        self.token2id = defaultdict(int)

        # prepare data
        self.token_set, train_data, test_data = self.load_data(data_dir, test_dir)


        # split data
        self.train_data, self.dev_data, self.test_data = self.split_data(train_data, test_data)

        # token and category vocabulary
        self.token2id = self.set2id(self.token_set, 'PAD', 'UNK')
        self.tag2id = self.set2id(set((train_data).keys()))

    def load_data(self, data_dir, test_dir):
        tok = spacy.load('en')
        label_file_path = ""
        text_file_path = ""
        train_data, test_data = defaultdict(list), defaultdict(list)
        token_set = set()

        #Split Test Directory into their target file and test file
        for path, dirnames, filenames in os.walk(test_dir):
            for file in filenames:
                #print('os.path.splittext: ', os.path.splitext(file))
                if os.path.splitext(file)[1] == '.csv':
                    file_path = path + "/" + file
                    label_file_path = file_path
                if os.path.splitext(file)[1] == '.tsv':
                    file_path = path + "/" + file
                    text_file_path = file_path

        #Training Data
        with open(data_dir) as data:
            rd = csv.reader(data, delimiter='\t', quotechar='"')
            for id, essaySet, score1, score2, essayText in rd:
                if id != 'Id':
                    line = [token.text for token in tok.tokenizer(essayText)]
                    train_data[score1].append(line)
                    for token in line:
                        token_set.add(token)
        # use dict to match the test data
        with open(text_file_path) as text_file, open(label_file_path) as label_file:
            label_file_csv = csv.reader(label_file, delimiter=',', quotechar='"')
            text_file_tsv = csv.reader(text_file, delimiter='\t', quotechar='"')

            labels = list(label_file_csv)
            texts = list(text_file_tsv)
            labels = labels[1:]
            texts = texts[1:]
            print(labels[0])
            print(texts[0])
            id2set, id2label, id2text = {}, {}, {}
            for item in labels:
                id2set[item[0]] = item[1]
            for item in labels:
                id2label[item[0]] = item[3]
            for item in texts:
                id2text[item[0]] = item[2]
            print("Number of test Labels : {}".format(len(id2set)))
            print("Number of test Texts : {}".format(len(id2text)))
            data = [] # [id, set, label, text]
            for k, v in id2label.items():
                item = []
                if k in id2text.keys():
                    item.append(k)
                    item.append(id2set[k])
                    item.append(v)
                    item.append(id2text[k])
                    data.append(item)
                    # print(item)
                    # assert 1==0

            print("Number of Matched data : {}".format(len(data)))

            for item in data:
                label = item[2]
                line = [token.text for token in tok.tokenizer(item[3])]
                test_data[label].append(line)




        return token_set, train_data, test_data

    def split_data(self, train_data, test_data):
        """
            Split data into train, dev, and test (currently use 80%/10%/10%)
            It is more make sense to split based on category, but currently it hurts performance
            """
        train_split = []
        dev_split = []

        print('Data statistics: ')

        all_data = []
        test = []
        print('Train statistics: ')
        for cat in train_data:
            cat_data = train_data[cat]
            print(cat, len(train_data[cat]))
            all_data += [(dat, cat) for dat in cat_data]
        print('Test statistics: ')
        for cat in test_data:
            cat_data = test_data[cat]
            print(cat, len(test_data[cat]))
            test += [(dat, cat) for dat in cat_data]
        all_data = random.sample(all_data, len(all_data))

        train_ratio = int(len(all_data) * 0.9)
        dev_ratio = int(len(all_data) * 1.0)

        train_split = all_data[:train_ratio]
        dev_split = all_data[train_ratio:dev_ratio]

        train_cat = set()
        for item, cat in train_split:
            train_cat.add(cat)
        print('Train categories:', sorted(list(train_cat)))

        dev_cat = set()
        for item, cat in dev_split:
            dev_cat.add(cat)
        print('Dev categories:', sorted(list(dev_cat)))

        test_cat = set()
        for item, cat in test:
            test_cat.add(cat)
        print('Test categories:', sorted(list(test_cat)))
        # assert 1==0

        return train_split, dev_split, test

    def set2id(self, item_set, pad=None, unk=None):
        item2id = defaultdict(int)
        if pad is not None:
            item2id[pad] = 0
        if unk is not None:
            item2id[unk] = 1

        for item in item_set:
            item2id[item] = len(item2id)

        return item2id

class PaddedTensorDataset(Dataset):
    """Dataset wrapping data, target and length tensors.
    Each sample will be retrieved by indexing both tensors along the first
    dimension.
    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        length (Tensor): contains sample lengths.
        raw_data (Any): The data that has been transformed into tensor, useful for debugging
    """

    def __init__(self, data_tensor, target_tensor, length_tensor, raw_data):
        assert data_tensor.size(0) == target_tensor.size(0) == length_tensor.size(0)
        self.data_tensor = data_tensor
        # print("=========================")
        # print(data_tensor)

        self.target_tensor = target_tensor
        self.length_tensor = length_tensor
        self.raw_data = raw_data


    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.length_tensor[index], self.raw_data[index]

    def __len__(self):
        return self.data_tensor.size(0)