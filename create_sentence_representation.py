import torch
import sys
import numpy as np
import scipy.io
import pickle
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

def save_to_mat(title, arr):
	firstlayer = np.array(arr[0])
	secondlayer = np.array(arr[1])
	scipy.io.savemat("embeddings/" + title + ".mat", mdict={'layer1': firstlayer, 'layer2': secondlayer})
	return

def save_to_pickle(title, arr):
	pickle.dump( arr, open("embeddings/" + title + ".p", "wb" ) )
	return

def main():
	# get input
	if len(sys.argv) != 2:
		print("Please enter prediction model.")
		exit()

	### GET MODEL VOCAB DICTIONARY
	model = sys.argv[1]
	without_pt = model.split(".")[0]
	print(without_pt)
	print(model)

	print('loading model...')
	vocab = torch.load(model)

	avg_sentence = process_sentence(vocab, "average")
	max_sentence = process_sentence(vocab, "maximum")
	min_sentence = process_sentence(vocab, "minimum")
	last_sentence = process_sentence(vocab, "last")

	### SAVE TO MAT FILE
	save_to_mat(without_pt + "-avg", avg_sentence)
	save_to_mat(without_pt + "-max", max_sentence)
	save_to_mat(without_pt + "-min", min_sentence)
	save_to_mat(without_pt + "-last", last_sentence)
	print("SAVED TO MATS")

	### SAVE TO PICKLE FILE
	save_to_pickle(without_pt + "-avg", avg_sentence)
	save_to_pickle(without_pt + "-max", max_sentence)
	save_to_pickle(without_pt + "-min", min_sentence)
	save_to_pickle(without_pt + "-last", last_sentence)
	print("SAVED TO PICKLES")


	return

if __name__ == "__main__":
    main()