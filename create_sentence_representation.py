import torch
import sys
import numpy as np
import scipy.io
import pickle
import os
from get_dict import get_missing, get_missing_counts, find_missing_sentences, update_dict
# import matlab

def process_sentence(vocab, typ):
	nsentences = len(vocab)
	print("number of sentences: " + str(nsentences))

	print('mixing sentences...')
	sentence_rep = [[], []]
	for sen in vocab:
		one_sent = [[], []]
		for token in sen:
			for layer in range(len(token)):
				one_sent[layer].append(np.array(token[layer]))

		for lay in range(len(one_sent)):
			if typ == "average":
				sentence_rep[lay].append(np.mean(one_sent[lay], axis = 0))
			elif typ == "maximum":
				sentence_rep[lay].append(np.array(one_sent[lay]).max(axis=0))
			elif typ == "minimum":
				sentence_rep[lay].append(np.array(one_sent[lay]).min(axis = 0))
			elif typ == "last":
				sentence_rep[lay].append(np.array(one_sent[lay][-1]))
				
	print("DOUBLE CHECK")
	print(len(sentence_rep))
	print(len(sentence_rep[0]))
	print(len(sentence_rep[0][0]))
	return sentence_rep

def save_to_mat(title, arr, bool_labels, method):

	print("\ncreating " + method + " matrix...")

	for layer_num in range(len(arr)):

		print("processing LAYER " + str(layer_num + 1) + "...")

		layer_dict = {}
		mat_title = "layer" + str(layer_num + 1) + "-" + str(method) + ".mat"

		for i in range(len(arr[layer_num])):
			sentence_label = "sentence" + str(i + 1)
			bool_label = bool_labels[i]
			embed = arr[layer_num][i]
			tup = [bool_label]
			tup.extend(embed)

			layer_dict[sentence_label] = tup

		scipy.io.savemat(title + "-" + mat_title, mdict=layer_dict)
	return

def save_to_pickle(title, arr):
	pickle.dump( arr, open("embeddings/" + title + ".p", "wb" ) )
	return

def get_missing_bools(model, file_name):
	get_dict_vocab = torch.load(model)
	print("IF ONLY 50k")
	top_50k_missing = get_missing(get_dict_vocab, file_name, top_50k = True)
	top_50k_missing_dict, top_50k_sentences = get_missing_counts(top_50k_missing, file_name)
	top_50k_sentences_with_missing, top_50k_missing_bools = find_missing_sentences(top_50k_missing_dict, top_50k_sentences, verbose = False)
	print("NUMBER OF TOP 50K MISSING SENTENCES:", len(top_50k_sentences_with_missing))
	return top_50k_missing_bools

def main():
	# get input
	if len(sys.argv) != 4:
		print("usage: python create_sentence_representation.py -sentences.txt -EXAMPLE.vocab.pt -EXAMPLE.pred.pt")
		exit()

	### GET MODEL VOCAB DICTIONARY
	saved_before = False
	sent_file_name = sys.argv[1]
	word_vocab = sys.argv[2]
	model = sys.argv[3]
	without_pt = model.split(".")[0]

	if not os.path.exists('embeddings/'):
		os.makedirs('embeddings/')

	print(without_pt)
	print(model)

	print('loading model...')
	vocab = torch.load(model)

	avg_sentence = process_sentence(vocab, "average")
	max_sentence = process_sentence(vocab, "maximum")
	min_sentence = process_sentence(vocab, "minimum")
	last_sentence = process_sentence(vocab, "last")

	
	bool_labels = get_missing_bools(word_vocab, sent_file_name)

	save_to_mat("embeddings/" + without_pt, avg_sentence, bool_labels, "avg")
	save_to_mat("embeddings/" + without_pt, max_sentence, bool_labels, "max")
	save_to_mat("embeddings/" + without_pt, min_sentence, bool_labels, "min")
	save_to_mat("embeddings/" + without_pt, last_sentence, bool_labels, "last")

	print("done.")

	return

if __name__ == "__main__":
    main()