import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class maLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, embedding_matrix):
        super(maLSTMClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.leftlstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        self.rightlstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0.2)

    def exponent_neg_manhattan_distance(self, x1, x2):
        return torch.exp(-torch.sum(torch.abs(x1-x2), dim=1))

    def init_hidden(self, batch_size):
        return torch.zeros(2,1,batch_size,self.hidden_dim)
        #return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
        #        autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, batch, lengths):
        self.hidden = self.init_hidden(batch.size(-1))

        embedsleft = self.embedding(batch[0])
        embedsright = self.embedding(batch[1])

        batch_size = embedsleft.size()[1]

        #packed_input_left = pack_padded_sequence(embedsleft, lengths[0])
        #packed_input_right = pack_padded_sequence(embedsright, lengths[1])

        outputsleft, (htLeft, ctLeft) = self.leftlstm(embedsleft, self.hidden)
        outputsright, (htRight, ctRight) = self.rightlstm(embedsright, self.hidden)

        output = self.exponent_neg_manhattan_distance(htLeft[0].permute(1,2,0).view(batch_size, -1),
                                                      htRight[0].permute(1,2,0).view(batch_size, -1))

        print('output: ', output)
        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        #output = self.dropout_layer(ht[-1])
        #output = self.hidden2out(output)
        #output = self.softmax(output)


        return output