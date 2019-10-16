import torch
import numpy as np
import pickle
import os
from get_dict import get_missing, get_missing_counts, find_missing_sentences, update_dict
import argparse
from tqdm import tqdm

def get_word_mapping(vocab, actual_words, num_layers):
	nsentences = len(vocab)
	word_map = {}
	print("number of sentences: " + str(nsentences))
	for sentence in tqdm(range(len(vocab))):
		one_sent = {}
		words_read = actual_words[sentence]
		for layer in range(num_layers):
			per_layer = {}
			word_counter = 0
			for word in vocab[sentence]:
				per_layer[words_read[word_counter]] = word[layer]
				word_counter += 1
			one_sent[layer] = per_layer
		word_map[sentence] = one_sent
	return word_map

def get_array_of_sentences(sent_file_name):
	text_file = open(vocab, "r")
	sentence_mapping = {}
	sentence_num = 0
	for line in text_file:
		sentence_mapping[sentence_num] = line.strip().split(" ")
		sentence_num+=1
	return sentence_mapping

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
	# if len(sys.argv) != 5:
	# 	print("usage: python save_individual_words.py -sentences.txt -EXAMPLE.vocab.pt -EXAMPLE.pred.pt -num_layers")
	# 	exit()

	argparser = argparse.ArgumentParser(description="save individual embeddings of words after training")
	argparser.add_argument('--vocab', type=str, help="Location of vocab.pt model after preprocessing", required=True)
	argparser.add_argument("--sentences", type=str, help=".txt file from fMRI experiment", required=True)
	argparser.add_argument("--prediction", type=str, help="Location of pred.pt model after prediction", required=True)
	argparser.add_argument("--num_layers", type=int, help="total number of layers in model", required=True)
	args = argparser.parse_args()

	### GET MODEL VOCAB DICTIONARY
	sent_file_name = args.sentence
	word_vocab = args.vocab
	model = args.prediction
	num_layers = args.num_layers
	without_pt = model.split("/")[-1].split(".")[0]

	if not os.path.exists('../word-embeddings/' + without_pt):
		os.makedirs('../word-embeddings/' + without_pt + '/' + )

	save_path = '../word-embeddings/' + without_pt + '/'

	print(without_pt)
	print(model)

	actual_words = get_array_of_sentences(sent_file_name)

	vocab = torch.load(model)
	word_embeddings = get_word_mapping(vocab, actual_words, num_layers)
	bool_labels = get_missing_bools(word_vocab, sent_file_name)

	pickle.dump( word_embeddings, open(save_path + "word-embeddings.p", "wb" ) )
	pickle.dump( bool_labels, open(save_path + "bool_labels.p", "wb" ) )

	print("done.")

	return

if __name__ == "__main__":
    main()