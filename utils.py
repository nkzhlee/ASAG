import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data import PaddedTensorDataset


def vectorized_data(data, item2id):
	return [[item2id[token] if token in item2id else item2id['UNK'] for token in seq] for (seq, _), _ in data]

def vectorized_refans(data, item2id):
	return [[[item2id[tok] if tok in item2id else item2id['UNK'] for tok in token] for token in x] for (_,x), _ in data]


def pad_sequences(vectorized_seqs, seq_lengths):
	# create a zero matrix
	seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()

	# fill the index
	for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
		seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

	#print("seq_tensor: ", seq_tensor)
	return seq_tensor

def pad_refAns(vectorized_seqs, seq_lengths, maxWords, maxRefAns):
	# create a zero matrix
	seq_tensor = torch.zeros((len(vectorized_seqs), maxRefAns, maxWords)).long()
	ref_lengths = torch.zeros((len(vectorized_seqs), maxRefAns)).long()

	#print('vectorized_seqs: ', vectorized_seqs)

	# fill the index
	for refAnsIdx, (seqs, lengths) in enumerate(zip(vectorized_seqs, seq_lengths)):

		ref_lengths[refAnsIdx, :len(lengths)] = torch.LongTensor(lengths)
		for wordIdx, (seq, seqlen) in enumerate(zip(seqs, lengths)):
			#print('seq: ', seq)
			#print('seqlen: ', seqlen)
			seq_tensor[refAnsIdx][wordIdx, :seqlen] = torch.LongTensor(seq)

	#print("seq_tensor: ", seq_tensor)
	return seq_tensor, ref_lengths


def create_dataset(data, input2id, target2id, batch_size=1):
	print("data.size(): ", len(data))
	vectorized_seqs = vectorized_data(data, input2id)
	vectorized_ref_ans = vectorized_refans(data,input2id)
	# print("++++++++++++")
	#print("vectorized_seqs: ", len(vectorized_seqs))
	#print("vectorized_seqs: ", vectorized_seqs)

	#num students * number of ref ans * num of words

	seq_lengths = torch.LongTensor([len(s) for s in vectorized_seqs])

	ref_ans_lengths = [[len(a) for a in s] for s in vectorized_ref_ans]
	#print('ref_ans_lengths: ', ref_ans_lengths)
	maxWordLength = max([max(s) for s in ref_ans_lengths])
	#print('maxWordLength: ', maxWordLength)
	maxRefAnsLength = max([len(s) for s in vectorized_ref_ans])
	#print('max number of reference answer: ', maxRefAnsLength)
	#ref_ans_lengths = torch.ragged.constant(b)
	#print("seq_lengths: ", seq_lengths)
	#print("seq_lengths.size(): ", seq_lengths.size())
	#print('ref_ans_lengths: ', ref_ans_lengths.size())

	ref_ans_tensor, ref_ans_lengths = pad_refAns(vectorized_ref_ans, ref_ans_lengths, maxWordLength, maxRefAnsLength)
	seq_tensor = pad_sequences(vectorized_seqs, seq_lengths)
	target_tensor = torch.LongTensor([target2id[y] for _, y in data])
	#print("seq_tensor.size(): ", seq_tensor.size())
	print("target_tensor: ", target_tensor)
	#print("target_tensor.size(): ", target_tensor.size())

	raw_data = [x for (x, _), _ in data]
	ref_ans = [x for (_,x), _ in data]

	#print("raw_data: ", raw_data)
	#print("fileID: ", fileId)

	return DataLoader(PaddedTensorDataset(seq_tensor, target_tensor, seq_lengths, raw_data, ref_ans_tensor, ref_ans_lengths), batch_size=batch_size)


def sort_batch(batch, targets, refAns, lengths, refAnsLengths):
	#print("batch: ", batch.size())
	#print("targets: ", targets)
	#print("lengths: ", lengths)
	#print('refAns: ', refAns.size())
	#print('refAnsLengths: ', refAnsLengths)

	seq_lengths, perm_idx = lengths.sort(0, descending=True)
	#print("seq_lengths: ", seq_lengths)
	#print("perm_idx: ", perm_idx)

	seq_tensor = batch[perm_idx]
	#print("seq_tensor: ", seq_tensor)
	#print("seq_tensor.size: ", seq_tensor.size())

	target_tensor = targets[perm_idx]
	#print("target_tensor.size: ", target_tensor.size())

	refAns_tensor = refAns[perm_idx]
	#print("target_tensor.size: ", refAns_tensor.size())

	refAnsLengths_tensor = refAnsLengths[perm_idx]

	return seq_tensor, target_tensor, seq_lengths, refAns_tensor, refAnsLengths_tensor
