import torch
import torch.autograd as autograd
import torch.nn as nn
#import torch.functional as F
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class textCNN(nn.Module):

	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, embedding_matrix):

		super(textCNN, self).__init__()



	def forward(self, batch, lengths):



		return output