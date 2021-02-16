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

        self.lstm_1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_2 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0.2)


    def exponent_neg_manhattan_distance(self, x1, x2):
        ''' Helper function for the similarity estimate of the LSTMs outputs '''
        return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))

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
        repeat_num = refAns.shape[1]
        refAnsLengths = refAnsLengths[0]
        print('studentAns: ', studentAns.size())
        print('lengths: ', lengths)
        print('refAns: ', refAns.size())
        print('refAnsLengths: ', refAnsLengths)
        studentAns = studentAns.view(-1, studentAns.shape[0], studentAns.shape[1])
        #print('studentAns: ', studentAns.size())
        studentAns = studentAns.repeat(1, repeat_num, 1)
        lengths = lengths.repeat(repeat_num)

        emb = self.embedding(studentAns)
        studentEmbed = emb.view(-1, emb.shape[2], emb.shape[3])
        print('studentEmbed: ', studentEmbed.size())

        emb = self.embedding(refAns)
        refEmbed = emb.view(-1, emb.shape[2], emb.shape[3])
        print('refEmbed: ', refEmbed.size())

        studentEmbedPack = pack_padded_sequence(
			studentEmbed, lengths, batch_first=True
		)

        print('studentEmbedPack: ', studentEmbedPack.data.size())

        refEmbedPack = pack_padded_sequence(
            refEmbed, refAnsLengths, batch_first=True
        )
        print('refEmbedPack: ', refEmbedPack.data.size())

        studentOutputs, (stu_ht, stu_ct) = self.lstm_1(studentEmbedPack)
        refAnsOutputs, (ref_ht, ref_ct) = self.lstm_2(refEmbedPack)
        print('stu_ht: ', stu_ht.size())
        print('ques_ht: ', ref_ht.size())

        stu_ht = stu_ht.view(stu_ht.shape[1], stu_ht.shape[2])
        ref_ht = ref_ht.view(ref_ht.shape[1], ref_ht.shape[2])

        print('stu_ht: ', stu_ht.size())
        print('ques_ht: ', ref_ht.size())

        # similarity_scores = self.exponent_neg_manhattan_distance(hidden_1[0].permute(1, 2, 0).view(batch_size, -1),
        #                                                          hidden_2[0].permute(1, 2, 0).view(batch_size, -1))

        similarity_scores = self.exponent_neg_manhattan_distance(stu_ht, ref_ht)
        print(similarity_scores)


        max = similarity_scores.max()
        print(max)

        assert 1 == 0
        return result