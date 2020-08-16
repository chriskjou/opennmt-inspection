import torch
import sys
import numpy as np
import scipy.io
import pickle
import os
from get_dict import get_missing, get_missing_counts, find_missing_sentences, update_dict
import argparse

def process_sentence(vocab, typ, num_layers):
	nsentences = len(vocab)
	print("number of sentences: " + str(nsentences))

	print('mixing sentences...')
	sentence_rep = [[] for _ in range(num_layers)]
	for sen in vocab:
		one_sent = [[] for _ in range(num_layers)]
		for token in sen:
			for layer in range(len(token)):
				one_sent[layer].append(np.array(token[layer]))

		for lay in range(len(one_sent)):
			if typ == "avg":
				sentence_rep[lay].append(np.mean(one_sent[lay], axis=0))
			elif typ == "max":
				sentence_rep[lay].append(np.array(one_sent[lay]).max(axis=0))
			elif typ == "min":
				sentence_rep[lay].append(np.array(one_sent[lay]).min(axis=0))
			elif typ == "last":
				sentence_rep[lay].append(np.array(one_sent[lay][-1]))
				
	# print("DOUBLE CHECK")
	# print(len(sentence_rep))
	# print(len(sentence_rep[0]))
	# print(len(sentence_rep[0][0]))
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
	argparser = argparse.ArgumentParser(description="create sentence representations for OpenNMT-py embeddings")
	argparser.add_argument("-num_layers", '--num_layers', type=int, default=4, help="num_layers")
	argparser.add_argument("-word_vocab", '--word_vocab', type=str, help="file path of the word model", required=True)
	argparser.add_argument("-model", '--model', type=str, help="file path of the prediction model", required=True)
	args = argparser.parse_args()

	### GET MODEL VOCAB DICTIONARY
	sent_file_name = "cleaned_examplesGLM.txt"
	without_pt = args.model.split("/")[-1].split(".")[0]

	methods = ['avg', 'max', 'min', 'last']

	if not os.path.exists('embeddings/' + without_pt):
		for method in methods:
			os.makedirs('embeddings/' + without_pt + '/' + method)

	print(without_pt)
	print(args.model)

	print('loading model...')
	vocab = torch.load(args.model)

	avg_sentence = process_sentence(vocab, "avg", args.num_layers)
	max_sentence = process_sentence(vocab, "max", args.num_layers)
	min_sentence = process_sentence(vocab, "min", args.num_layers)
	last_sentence = process_sentence(vocab, "last", args.num_layers)

	mats = [avg_sentence, max_sentence, min_sentence, last_sentence]
	
	bool_labels = get_missing_bools(args.word_vocab, sent_file_name)

	for i in range(len(methods)):
		save_to_mat("embeddings/" + without_pt + '/' + methods[i] + '/', mats[i], bool_labels, methods[i])

	print("done.")

	return

if __name__ == "__main__":
    main()