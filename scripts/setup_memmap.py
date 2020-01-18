from tqdm import tqdm
import pickle
import numpy as np
import sys
import math
import argparse
import os
import helper

def get_dimensions():
	return

def create_memmap(args, file_path, temp_file_name, num_voxels, a, b):
	if args.brain_to_model:
		size = (a,b)
	else:
		size = (b,a)

	fp = np.memmap(file_path + temp_file_name + ".dat", dtype='float32', mode='w+', shape=(num_voxels, size[0], size[1]))
	del fp


def main():
	argparser = argparse.ArgumentParser(description="create memmap for decoding")
	argparser.add_argument('--embedding_layer', type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("--subject_mat_file", type=str, help=".mat file ")
	argparser.add_argument("--brain_to_model", action='store_true', default=False, help="True if regressing brain to model, False if not")
	argparser.add_argument("--model_to_brain", action='store_true', default=False, help="True if regressing model to brain, False if not")
	argparser.add_argument("--cross_validation", action='store_true', default=False, help="True if add cross validation, False if not")
	argparser.add_argument("--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	argparser.add_argument("--batch_num", type=int, help="batch number of total (for scripting) (out of --total_batches)", required=True)
	argparser.add_argument("--total_batches", type=int, help="total number of batches", required=True)
	argparser.add_argument("--random",  action='store_true', default=False, help="True if initialize random brain activations, False if not")
	argparser.add_argument("--rand_embed",  action='store_true', default=False, help="True if initialize random embeddings, False if not")
	argparser.add_argument("--glove",  action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	argparser.add_argument("--word2vec",  action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("--bert",  action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("--normalize",  action='store_true', default=False, help="True if add normalization across voxels, False if not")
	argparser.add_argument("--permutation",  action='store_true', default=False, help="True if permutation, False if not")
	argparser.add_argument("--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	args = argparser.parse_args()

	create_memmap(args, "/n/shieber_lab/Lab/users/cjou/predictions_od32/", num_voxels, predicted_size)
	create_memmap(args, "/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/", num_voxels, true_size)
	print("done.")
	return

if __name__ == "__main__":
	main()