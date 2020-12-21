import torch
import torch.autograd as autograd
import torch.nn as nn
#import torch.functional as F
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LogisticRegression(nn.Module):

	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, embedding_matrix):

		super(LogisticRegression, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size

		#self.embedding = nn.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])

		self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))

		#self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

		#self.hidden2out = nn.Linear(hidden_dim, output_size)
		self.softmax = nn.LogSoftmax()
		self.logreg = nn.Linear(embedding_dim, output_size)

		#self.dropout_layer = nn.Dropout(p=0.2)



	#def init_hidden(self, batch_size):
	#	return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
	#					autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))


	def forward(self, batch, lengths):

		print("batch[0]: ", batch[0])
		print("batch size: ", batch.size())
		print("lengths: ", lengths)

		embeds = self.embedding(batch)

		#print("embeds: ", embeds)
		print("embeds: ", embeds.size())

		packed_input = pack_padded_sequence(embeds, lengths) #to keep consistent vector

		#print("packed_input: ", packed_input)
		print("packed_input.data.size: ", packed_input.data.size())
		print("packed_input.batch_sizes.size: ", packed_input.batch_sizes.size())

		#Does this feed both the packed input tensor vectors along with the batch lengths?
		#I get an error running this, so I wonder if I have to just pass in the passed input
		#tensor vector (packed_input.data instead of just packed_input)

		linear = self.logreg(packed_input.data)

		print("linear.size: ", linear.size())

		output = F.sigmoid(linear)
		#print('output: ', output)
		print('output.size: ', output.size())
		output = self.softmax(output)
		#print('output: ', output)
		print('output: ', output.size())

		return output