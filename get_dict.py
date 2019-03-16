import torch
import sys
from collections import Counter
import operator

def get_missing(vocab, top_50k = False):
	src_vocab = vocab[0][1]
	print("SRC VOCAB: " + str(len(src_vocab)))
	tgt_vocab = vocab[1][1]
	print("TG VOCAB: " + str(len(tgt_vocab)))

	### GET TEST VOCAB SET
	wordcount = Counter()
	test_file = "cleaned_sentencesGLM.txt"
	with open(test_file) as file:
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

def get_missing_counts(missing):
	### GET GLM VOCAB
	file = open("cleaned_sentencesGLM.txt","r")

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
	if len(sys.argv) != 2:
		print("usage: python get_dict.py -EXAMPLE.vocab.pt")
		exit()

	### GET MODEL VOCAB DICTIONARY
	model = sys.argv[1]
	print(model)

	print("ALL")
	vocab = torch.load(model)
	missing = get_missing(vocab)
	missing_dict, sentences = get_missing_counts(missing)
	sentences_with_missing, missing_bools = find_missing_sentences(missing_dict, sentences)
	print("NUMBER OF MISSING SENTENCES:", len(sentences_with_missing))
	print(sentences_with_missing)
	print()

	print("IF ONLY 50k")
	top_50k_missing = get_missing(vocab, top_50k = True)
	top_50k_missing_dict, top_50k_sentences = get_missing_counts(top_50k_missing)
	top_50k_sentences_with_missing, top_50k_missing_bools = find_missing_sentences(top_50k_missing_dict, top_50k_sentences)
	print("NUMBER OF TOP 50K MISSING SENTENCES:", len(top_50k_sentences_with_missing))
	print(top_50k_sentences_with_missing)
	return

if __name__ == "__main__":
    main()