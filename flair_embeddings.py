from flair.embeddings import BertEmbeddings
from flair.data import Sentence
import numpy as np
import os
import pickle

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

def get_bert_embeddings(file, model):
	embed_matrix = []
	for line in file:
		sentence = Sentence(line)
		model.embed(sentence)
		vector = [token.embedding.numpy() for token in sentence]
		embed_matrix.append(np.array(vector))
	return embed_matrix

def main():
	file = open("cleaned_sentencesGLM.txt","r").read().splitlines()

	print("uploading bert model...")
	bert_embedding = BertEmbeddings("bert-base-multilingual-cased")

	embed_matrix = get_bert_embeddings(file, bert_embedding)

	print("aggregating types...")
	avg_sentence = process_sentence(embed_matrix, "avg")
	max_sentence = process_sentence(embed_matrix, "max")
	min_sentence = process_sentence(embed_matrix, "min")
	last_sentence = process_sentence(embed_matrix, "last")

	methods = ['avg', 'max', 'min', 'last']
	mats = [avg_sentence, max_sentence, min_sentence, last_sentence]

	bool_labels = [1]*len(file)

	if not os.path.exists('../embeddings/bert/'):
		os.makedirs('../embeddings/bert/')

	print("saving files...")
	for i in range(len(methods)):
		pickle.dump(mats[i], open("../embeddings/bert/" + str(methods[i]) + ".p", "wb"))


	print("done.")

if __name__ == "__main__":
	main()