import torch
import sys
from collections import Counter
import operator
import argparse

def get_missing(vocab, file_name, top_50k = False):
	src_vocab = vocab[0][1]
	print("SRC VOCAB: " + str(len(src_vocab)))
	tgt_vocab = vocab[1][1]
	print("TG VOCAB: " + str(len(tgt_vocab)))

	### GET TEST VOCAB SET
	wordcount = Counter()
	with open(file_name) as file:
	    for line in file:
	        wordcount.update(line.lower().split())

	### GET VOCAB SETS
	if top_50k:
		frqs = src_vocab.freqs
		training_vocab = set(dict(sorted(frqs.items(), key=operator.itemgetter(1), reverse = True)[:50000]).keys())
	else:
		training_vocab = set(src_vocab.freqs.keys())
	test_vocab = set(wordcount.keys())

	missing = test_vocab.difference(training_vocab)
	print("MISSING WORDS FROM TEST DATA SET: ")
	print(missing)
	print("NUMBER OF MISSING: " + str(len(missing)))
	return missing

def get_missing_counts(missing, file_name):
	### GET GLM VOCAB
	file = open(file_name,"r")

	c = Counter()
	sentences = {}

	count = 0 
	for line in file:
		line = line.replace("_", " ")
		rem_n = line.rstrip()
		line = rem_n.lower()
		c.update(line.split(" "))
		sentences[count] = line
		count+=1

	# print(c)
	print("SIZE OF DICTIONARY: " + str(len(c)))

	# get counts of missing words
	words_keys = set(missing)
	all_keywords = set(c)
	intersection = words_keys & all_keywords

	missing_dict = { k: c[k] for k in intersection }
	print("COUNTS FOR MISSING WORDS:")
	print(missing_dict)
	print("NUMBER OF TOTAL MISSING WORD APPEARANCES:")
	print(sum(missing_dict.values()))
	return missing_dict, sentences

def find_missing_sentences(missing_dict, sentences, verbose = True):
	missing_words = set(missing_dict.keys())
	sentences_with_missing = []
	missing_bools = []
	for key, value in sentences.items():
		words = value.split(" ")
		inter = missing_words.intersection(words)
		if len(missing_words.intersection(words)) != 0:
			sentences_with_missing.append(value)
			missing_bools.append(0)
			if verbose:
				print("MISSING:", inter)
				print("SENTENCE #", key, ": ", value)
				print()
		else:
			missing_bools.append(1)
	return sentences_with_missing, missing_bools

def update_dict():
	top_50 = dict(sorted(freq.items(), key=operator.itemgetter(1), reverse = True)[:50000])
	return

def main():
	if len(sys.argv) != 3:
		print("usage: python get_dict.py -sentences.txt -EXAMPLE.vocab.pt")
		exit()

	argparser = argparse.ArgumentParser(description="create sentence representations for OpenNMT-py embeddings")
	argparser.add_argument("-model", '--model', type=str, help="file path of the prediction model", required=True)
	args = argparser.parse_args()

	### GET MODEL VOCAB DICTIONARY
	sent_file_name = "cleaned_examplesGLM.txt"
	print(args.model)

	print("ALL")
	vocab = torch.load(args.model)
	missing = get_missing(vocab, sent_file_name)
	missing_dict, sentences = get_missing_counts(missing, sent_file_name)
	sentences_with_missing, missing_bools = find_missing_sentences(missing_dict, sentences)
	print("NUMBER OF MISSING SENTENCES:", len(sentences_with_missing))
	print(sentences_with_missing)
	print()

	print("IF ONLY 50k")
	top_50k_missing = get_missing(vocab, sent_file_name, top_50k = True)
	top_50k_missing_dict, top_50k_sentences = get_missing_counts(top_50k_missing, sent_file_name)
	top_50k_sentences_with_missing, top_50k_missing_bools = find_missing_sentences(top_50k_missing_dict, top_50k_sentences)
	print("NUMBER OF TOP 50K MISSING SENTENCES:", len(top_50k_sentences_with_missing))
	print(top_50k_sentences_with_missing)
	return

if __name__ == "__main__":
    main()