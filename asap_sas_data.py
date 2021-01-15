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

class ASAPSASTextLoader:
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
        target_file = 0
        test_file = 0
        train_data, test_data = defaultdict(list), defaultdict(list)
        token_set = set()

        #Split Test Directory into their target file and test file
        for path, dirnames, filenames in os.walk(test_dir):
            for file in filenames:
                #print('os.path.splittext: ', os.path.splitext(file))
                if os.path.splitext(file)[1] == '.csv':
                    file_path = path + "/" + file
                    target_file = file_path
                if os.path.splitext(file)[1] == '.tsv':
                    file_path = path + "/" + file
                    test_file = file_path

        #Training Data
        with open(data_dir) as data:
            rd = csv.reader(data, delimiter='\t', quotechar='"')
            for id, essaySet, score1, score2, essayText in rd:
                if id != 'Id':
                    line = [token.text for token in tok.tokenizer(essayText)]
                    train_data[score1].append(line)
                    for token in line:
                        token_set.add(token)

        #Test Data
        #Iterate through target and test file in parallel and combine them into a dict
        with open(test_file) as test, open(target_file) as target:
            rd1 = csv.reader(test, delimiter='\t', quotechar='"')
            rd2 = csv.reader(target, delimiter=',', quotechar='"')
            rd1 = list(rd1)
            rd2 = list(rd2)
            targetIter = 1
            for testIter in range(len(rd1)):
                if rd1[testIter][0] != 'Id':
                    #print('rd2: ', rd2[iterator])
                    #print('rd1: ', rd1[iterator])
                    idTest,idTarget = rd1[testIter][0], rd2[targetIter][0]


                    #Because some solutions are missing only match the scores with the same ID
                    if idTest == idTarget:
                        #print('idTest, idTarget: ', (idTest, idTarget))
                        essayScore, essaySet, essayText = rd2[targetIter][3], rd1[testIter][1], rd1[testIter][2]

                        line = [token.text for token in tok.tokenizer(essayText)]
                        test_data[essayScore].append(line)

                        #only iterate target when there is a match
                        targetIter += 1
                        #for token in line:
                        #    token_set.add(token)



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

        for cat in train_data:
            cat_data = train_data[cat]
            print(cat, len(train_data[cat]))
            all_data += [(dat, cat) for dat in cat_data]

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