from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import BertEmbeddings, RoBERTaEmbeddings, XLMEmbeddings
from flair.data import Sentence
import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm

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
	argparser.add_argument("-num_layers", "--num_layers", type=int, help="num_layers", required=True)
	argparser.add_argument("-bert", "--bert", action='store_true', default=False, help="bert embeddings (0-12 layers)")
	argparser.add_argument("-roberta", "--roberta", action='store_true', default=False, help="roberta embeddings (0-12 layers)")
	argparser.add_argument("-gpt2", "--gpt2", action='store_true', default=False, help="gpt2 embeddings (0-12 layers)")
	argparser.add_argument("-xlm", "--xlm", action='store_true', default=False, help="xlm embeddings (0-24 layers)")
	argparser.add_argument("-local", "--local", action='store_true', default=False, help="if local")
	args = argparser.parse_args()

	# verify arguments
	if args.bert and args.roberta and args.xlm and args.gpt2:
		print("select only one flag for model type from (bert, roberta, xlm)")
		exit()
	if not args.bert and not args.roberta and not args.xlm and not args.gpt2:
		print("select at least flag for model type from (bert, roberta, xlm)")
		exit()

	if args.bert and args.num_layers not in range(13):
		print("not a valid layer for bert. choose between 0-12 layers")
		exit()
	if args.roberta and args.num_layers not in range(13):
		print("not a valid layer for roberta. choose between 0-12 layers")
		exit()
	if args.xlm and args.num_layers not in range(25):
		print("not a valid layer for xlm. choose between 0-24 layers")
		exit()
	if args.gpt2 and args.num_layers not in range(49):
		print("not a valid layer for gpt2. choose between 0-48 layers")
		exit()

	# open sentences
	file = open("cleaned_sentencesGLM.txt","r").read().splitlines()

	# specify model
	print("uploading model...")
	for layer in [1]: # tqdm(range(args.num_layers)):
		print(layer)
		if args.bert:
			embeddings = BertEmbeddings("bert-base-multilingual-cased", layers="-{}".format(layer))
			model_type = "bert"
		elif args.roberta:
			embeddings = RoBERTaEmbeddings("roberta-base", layers="-{}".format(layer))
			model_type = "roberta"
		elif args.xlm:
			embeddings = XLMEmbeddings("xlm-mlm-en-2048", layers="-{}".format(layer))
			model_type = "xlm"
		elif args.gpt2:
			embeddings = TransformerWordEmbeddings("gpt2-xl", layers="-{}".format(layer))
			model_type = "gpt2"
		else:
			print("error on calling embeddings")
			exit()

		embed_matrix = get_embeddings(file, embeddings)

		print("aggregating types...")
		avg_sentence = process_sentence(embed_matrix, "avg")
		# max_sentence = process_sentence(embed_matrix, "max")
		# min_sentence = process_sentence(embed_matrix, "min")
		# last_sentence = process_sentence(embed_matrix, "last")

		methods = ['avg', 'max', 'min', 'last']
		mats = [avg_sentence, max_sentence, min_sentence, last_sentence]

		bool_labels = [1]*len(file)

		print("saving files...")
		if args.local:
			file_path = '../embeddings/{}/layer{}/'.format(model_type, layer)
		else:
			file_path = "/n/shieber_lab/Lab/users/cjou//embeddings/{}/layer{}/".format(model_type, layer)

		if not os.path.exists(file_path):
			os.makedirs(file_path)

		for i in range(len(methods)):
			print("saving file: " + file_path + str(methods[i]) + ".p")
			pickle.dump(mats[i], open(file_path + str(methods[i]) + ".p", "wb"))

	print("done.")

if __name__ == "__main__":
	main()