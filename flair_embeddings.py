from flair.embeddings import BertEmbeddings
from flair.data import Sentence
import numpy as np
import os
import pickle
import argparse

def process_sentence(vocab, typ):
	nsentences = len(vocab)
	print("number of sentences: " + str(nsentences))
	print('mixing sentences...')
	sentence_rep = []
	for i in range(nsentences):
		if typ == "avg":
			sentence_rep.append(np.mean(vocab[i], axis=0))
		if typ == "max":
			sentence_rep.append(np.array(vocab[i]).max(axis=0))
		elif typ == "min":
			sentence_rep.append(np.array(vocab[i]).min(axis=0))
		elif typ == "last":
			sentence_rep.append(np.array(vocab[i])[-1])
	print("DOUBLE CHECK")
	print(len(sentence_rep))
	print(len(sentence_rep[0]))
	return sentence_rep

def get_embeddings(file, model):
	embed_matrix = []
	for line in file:
		sentence = Sentence(line)
		model.embed(sentence)
		vector = [token.embedding.numpy() for token in sentence]
		embed_matrix.append(np.array(vector))
	return embed_matrix

def main():
	argparser = argparse.ArgumentParser(description="download embeddings for models")
	argparser.add_argument("-embedding_layer", "--embedding_layer", type=int, help="embedding layer number", required=True)
	argparser.add_argument("-bert", "--bert", action='store_true', default=False, help="bert embeddings (0-12 layers)")
	argparser.add_argument("-roberta", "--roberta", action='store_true', default=False, help="roberta embeddings (0-12 layers)")
	argparser.add_argument("-xlm", "--xlm", action='store_true', default=False, help="xlm embeddings (0-24 layers)")
	args = argparser.parse_args()

	# verify arguments
	if args.bert and args.roberta and args.xlm:
		print("select only one flag for model type from (bert, roberta, xlm)")
		exit()
	if not args.bert and not args.roberta and not args.xlm:
		print("select at least flag for model type from (bert, roberta, xlm)")
		exit()

	if args.bert and args.embedding_layer not in range(12):
		print("not a valid layer for bert. choose between 0-12 layers")
	if args.roberta and args.embedding_layer not in range(12):
		print("not a valid layer for roberta. choose between 0-12 layers")
	if args.xlm and args.embedding_layer not in range(24):
		print("not a valid layer for xlm. choose between 0-24 layers")

	# open sentences
	file = open("cleaned_sentencesGLM.txt","r").read().splitlines()

	# specify model
	print("uploading model...")
	if args.bert:
		embeddings = BertEmbeddings("bert-base-multilingual-cased", layers="-{}".format(args.embedding_layer))
		model_type = "bert"
	elif args.roberta:
		embeddings = RoBERTaEmbeddings("roberta-base", layers="-{}".format(args.embedding_layer))
		model_type = "roberta"
	elif args.xlm:
		embeddings = XLMEmbeddings("xlm-mlm-en-2048", layers="-{}".format(args.embedding_layer))
		model_type = "xlm"
	else:
		print("error on calling embeddings")
		exit()

	embed_matrix = get_embeddings(file, embeddings)

	print("aggregating types...")
	avg_sentence = process_sentence(embed_matrix, "avg")
	max_sentence = process_sentence(embed_matrix, "max")
	min_sentence = process_sentence(embed_matrix, "min")
	last_sentence = process_sentence(embed_matrix, "last")

	methods = ['avg', 'max', 'min', 'last']
	mats = [avg_sentence, max_sentence, min_sentence, last_sentence]

	bool_labels = [1]*len(file)

	if not os.path.exists('../embeddings/{}/'.format(model_type)):
		os.makedirs('../embeddings/{}/'.format(model_type))

	print("saving files...")
	for i in range(len(methods)):
		pickle.dump(mats[i], open("../embeddings/{}/".format(model_type) + str(methods[i]) + ".p", "wb"))


	print("done.")

if __name__ == "__main__":
	main()