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

		#print("batch[0]: ", batch[0])
		#print("batch size: ", batch.size())
		#print("lengths: ", lengths)

		#print("batch: ", batch)
		#print("batch.size: ", batch.size())
		embeds = self.embedding(batch)

		#print("embeds: ", embeds)
		#print("embeds.size(): ", embeds.size())

		#Transpose matrix so that it becomes 64x50x82 and find average of third dimension
		#The resulting matrix should be 64x50

		#sentenceEmbed = self.sentenceEmbedding(embeds)
		#print("sentenceEmbed: ", len(sentenceEmbed))
		#print("sentenceEmbed: ", len(sentenceEmbed[0]))

		#sentenceEmbed = torch.FloatTensor(sentenceEmbed)
		#print("sentenceEmbed: ", sentenceEmbed)
		#print("sentenceEmbed: ", sentenceEmbed.size())

		sentenceEmbed = torch.mean(embeds,dim=1)
		#print("sentenceEmbeds: ", sentenceEmbed)
		#print("sentenceEmbed.size(): ", sentenceEmbed.size())

		#print('embeds[0][0]: ', embeds[0][0])
		#print('average of embeds[0][0]: ', sum(embeds[0][0])/len(embeds[0][0]))

		#print("embeds: ", embeds)
		#print("embeds: ", embeds.size())

		#packed_input = pack_padded_sequence(embeds, lengths) #to keep consistent vector

		#print("packed_input: ", packed_input)
		#print("packed_input.data.size: ", packed_input.data.size())
		#print("packed_input.batch_sizes.size: ", packed_input.batch_sizes.size())

		linear = self.logreg(sentenceEmbed)

		#print("linear.size: ", linear.size())

		output = F.sigmoid(linear)
		#print('output: ', output)
		#print('output.size: ', output.size())

		output = self.softmax(output)
		#print('output: ', output)
		#print('output: ', output.size())

		return output