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

        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))

        self.lstm_1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        self.lstm_2 = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0.2)

    def exponent_neg_manhattan_distance(self, x1, x2):
        return torch.exp(-torch.sum(torch.abs(x1-x2), dim=1))

    def init_hidden(self, batch_size):
        #return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
        #        autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))
        return (torch.zeros(1,batch_size,self.hidden_dim),
                torch.zeros(1,batch_size,self.hidden_dim))

    def init_weights(self):
        ''' Initialize weights of lstm 1 '''
        for name_1, param_1 in self.lstm_1.named_parameters():
            if 'bias' in name_1:
                nn.init.constant_(param_1, 0.0)
            elif 'weight' in name_1:
                nn.init.xavier_normal_(param_1)

        ''' Set weights of lstm 2 identical to lstm 1 '''
        lstm_1 = self.lstm_1.state_dict()
        lstm_2 = self.lstm_2.state_dict()

        for name_1, param_1 in lstm_1.items():
            # Backwards compatibility for serialized parameters.
            if isinstance(param_1, torch.nn.Parameter):
                param_1 = param_1.data

            lstm_2[name_1].copy_(param_1)

    def forward(self, studentAns, lengths, refAns, refAnsLengths):
        #studentAns: two dimension (batch size * word sequence)
        #refAns: three dimension (batch size * num of reference answers * word sequence)

        hidden = self.init_hidden(len(studentAns.data))

        print('studentAns: ', studentAns.size())
        print('lengths: ', len(lengths))
        print('refAns: ', refAns.size())
        print('refAnsLengths: ', len(refAnsLengths))

        studentEmbed = self.embedding(studentAns)
        #print('studentEmbed: ', studentEmbed)
        print('studentEmbed: ', studentEmbed.size())

        refEmbed = self.embedding(refAns)
        #print('studentEmbed: ', refEmbed)
        print('refEmbed: ', refEmbed.size())

        studentEmbedPack = pack_padded_sequence(
			studentEmbed, lengths, batch_first=True
		)
        print('studentEmbedPack: ', studentEmbedPack.data.size())

        refEmbedPack = pack_padded_sequence(
            refEmbed, lengths, batch_first=True
        )
        print('refEmbedPack: ', refEmbedPack.data.size())

        studentOutputs, (stu_ht, stu_ct) = self.lstm_1(studentEmbedPack, hidden)
        refAnsOutputs, (ques_ht, ques_ct) = self.lstm_2(refEmbedPack, hidden)

        print('stu_ht: ', stu_ht.size())
        print('ques_ht: ',ques_ht.size())


        return output