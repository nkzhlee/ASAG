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

		# init LSTM layer
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

		self.logreg = nn.Sequential(
			nn.Linear(hidden_dim, output_size),
		)


	def forward(self, batch, lengths):


		embeds = self.embedding(batch)

		# pack LSTM input
		# why? see this link: https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
		embed_pack = pack_padded_sequence(
			embeds, lengths, batch_first=True
		)
		outputs, (ht, ct) = self.lstm(embed_pack)
		# use hidden state ht as result
		output = self.logreg(ht.view(-1, self.hidden_dim))

		output = F.sigmoid(output)


		return output