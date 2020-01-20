import scipy.io
import pickle
import numpy as np
import sys
import argparse
import os
import helper

def get_embed_matrix(embedding, num_sentences=240):
	embed_matrix = np.array([embedding["sentence" + str(i+1)][0][1:] for i in range(num_sentences)])
	in_training_bools = np.array([embedding["sentence" + str(i+1)][0][0] for i in range(num_sentences)])
	return embed_matrix

def get_dimensions(args, activations, embed_matrix):
	ACTIVATION_SHAPE = (activations.shape[0], 515)

	if args.brain_to_model:
		target_size = embed_matrix.shape
	else:
		target_size = ACTIVATION_SHAPE

	VOXEL_NUMBER = activations.shape[1]
	return VOXEL_NUMBER, target_size[0], target_size[1]

def create_entire_memmap(file_path, dim1, dim2, dim3):
	global temp_file_name
	memmap = np.memmap(file_path + temp_file_name + ".dat", dtype='float32', mode='w+', shape=(dim1+1, dim2, dim3))
	memmap_size = np.zeros((1, dim2, dim3))
	memmap_size[0][0][0] = dim1
	memmap_size[0][0][1] = dim2
	memmap_size[0][0][2] = dim3
	memmap[0] = memmap_size	
	del memmap

def main():
	global temp_file_name

	argparser = argparse.ArgumentParser(description="Decoding (linear reg). step for correlating NN and brain")
	argparser.add_argument('--embedding_layer', type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("--subject_mat_file", type=str, help=".mat file ")
	argparser.add_argument("--brain_to_model", action='store_true', default=False, help="True if regressing brain to model, False if not")
	argparser.add_argument("--model_to_brain", action='store_true', default=False, help="True if regressing model to brain, False if not")
	argparser.add_argument("--cross_validation", action='store_true', default=False, help="True if add cross validation, False if not")
	argparser.add_argument("--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	argparser.add_argument("--random",  action='store_true', default=False, help="True if initialize random brain activations, False if not")
	argparser.add_argument("--rand_embed",  action='store_true', default=False, help="True if initialize random embeddings, False if not")
	argparser.add_argument("--glove",  action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	argparser.add_argument("--word2vec",  action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("--bert",  action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("--normalize",  action='store_true', default=False, help="True if add normalization across voxels, False if not")
	argparser.add_argument("--permutation",  action='store_true', default=False, help="True if permutation, False if not")
	argparser.add_argument("--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	argparser.add_argument("--memmap",  action='store_true', default=False, help="True if memmep, False if not")
	args = argparser.parse_args()

	if not args.glove and not args.word2vec and not args.bert and not args.rand_embed:
		embed_loc = args.embedding_layer
		file_name = embed_loc.split("/")[-1].split(".")[0]
		embedding = scipy.io.loadmat(embed_loc)
		embed_matrix = get_embed_matrix(embedding)
	else:
		embed_loc = args.embedding_layer
		file_name = embed_loc.split("/")[-1].split(".")[0].split("-")[-1] # aggregation type
		if args.word2vec:
			# embed_matrix = pickle.load( open( "../embeddings/word2vec/" + str(file_name) + ".p", "rb" ) )	
			embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/word2vec/" + str(file_name) + ".p", "rb" ) )	
		elif args.glove:
			# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
			embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/glove/" + str(file_name) + ".p", "rb" ) )	
		elif args.bert:
			# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
			embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/bert/" + str(file_name) + ".p", "rb" ) )
		else: # args.rand_embed
			# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
			embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/rand_embed/rand_embed.p", "rb" ) )	
	
	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)
	temp_file_name = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj" + str(args.subject_number) + "-" + str(file_name)
	activations = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/fmri/subj{}/".format(args.subject_number) + str(plabel) + str(prlabel) + "activations.p", "rb" ) )

	# get dimensions
	VOXEL_NUMBER, num_sentences, dim = get_dimensions(args, activations, embed_matrix)
	print("DIMENSIONS: " + str(VOXEL_NUMBER) + " " + str(num_sentences) + " " + str(dim))
	create_entire_memmap("/n/shieber_lab/Lab/users/cjou/predictions_memmap/", VOXEL_NUMBER, num_sentences, dim)
	if args.model_to_brain:
		create_entire_memmap("/n/shieber_lab/Lab/users/cjou/true_spotlights_memmap/", VOXEL_NUMBER, num_sentences, dim)
	print("done.")

	return

if __name__ == "__main__":
	main()