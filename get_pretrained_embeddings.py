import gensim
import pickle
import numpy as np
import os
import scipy.io

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

def get_missing_bools(file, model):
	missing_bools = []
	for line in file:
		words = [x for x in line.split(' ') if x not in model]
		if len(words) != 0:
			missing_bools.append(1)
		else:
			missing_bools.append(0)
	return missing_bools

def get_word2vec(file):
	print("WORD2VEC")

	print("loading model...")
	model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)

	print("getting representations...")
	file = open("cleaned_sentencesGLM.txt","r").read().splitlines()

	embed_matrix = []
	for line in file:
		# missing for 'to' and 'a'
		vectors = [model[x] for x in line.split(' ') if x in model]
		embed_matrix.append(np.array(vectors))

	print("dumping individual words...")
	pickle.dump( embed_matrix, open("word2vec_pretrained_embeddings_individual_words.p", "wb" ) )

	# embed_matrix = pickle.load( open( "word2vec_pretrained_embeddings_individual_words.p", "rb" ) )
	print("dumping sentence representations...")
	avg_sentence = process_sentence(embed_matrix, "avg")
	max_sentence = process_sentence(embed_matrix, "max")
	min_sentence = process_sentence(embed_matrix, "min")
	last_sentence = process_sentence(embed_matrix, "last")

	methods = ['avg', 'max', 'min', 'last']
	mats = [avg_sentence, max_sentence, min_sentence, last_sentence]
	
	bool_labels = get_missing_bools(file, model)

	if not os.path.exists('../embeddings/word2vec/'):
		os.makedirs('../embeddings/word2vec/')

	for i in range(len(methods)):
		pickle.dump(mats[i], open("../embeddings/word2vec/" + str(methods[i]) + ".p", "wb"))

	print("done.")
	return

def get_glove(file):
	print("GLoVE")

	print("loading model...")
	glove = open("../glove.6B/glove.6B.300d.txt","r").read().splitlines()

	model = {}
	for line in glove:
		name = line.split(" ")[0]
		value = np.array(line.split(" ")[1:]).astype(np.float)
		model[name] = value

	print("getting representations...")
	
	embed_matrix = []
	for line in file:
		# missing for 'to' and 'a'
		vectors = [model[x] for x in line.split(' ') if x in model]
		embed_matrix.append(np.array(vectors))

	print("dumping individual words...")
	pickle.dump( embed_matrix, open("glove_pretrained_embeddings_individual_words.p", "wb" ) )

	# embed_matrix = pickle.load( open( "word2vec_pretrained_embeddings_individual_words.p", "rb" ) )
	print("dumping sentence representations...")
	avg_sentence = process_sentence(embed_matrix, "avg")
	max_sentence = process_sentence(embed_matrix, "max")
	min_sentence = process_sentence(embed_matrix, "min")
	last_sentence = process_sentence(embed_matrix, "last")

	methods = ['avg', 'max', 'min', 'last']
	mats = [avg_sentence, max_sentence, min_sentence, last_sentence]
	
	bool_labels = get_missing_bools(file, model)

	if not os.path.exists('../embeddings/glove/'):
		os.makedirs('../embeddings/glove/')

	for i in range(len(methods)):
		pickle.dump(mats[i], open("../embeddings/glove/" + str(methods[i]) + ".p", "wb"))

	print("done.")
	return

def main():
	file = open("cleaned_sentencesGLM.txt","r").read().splitlines()
	# get_word2vec(file)
	get_glove(file)

	return

if __name__ == "__main__":
    main()
