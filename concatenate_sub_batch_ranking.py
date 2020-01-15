import os
import sys
import argparse
from tqdm import tqdm
import pickle
import helper

def find_files(args):

	# file name assignments
	if args.brain_to_model:
		dflag = " --brain_to_model"
	else:
		dflag = " --model_to_brain"

	if args.cross_validation:
		cvflag = " --cross_validation"
	else:
		cvflag = ""

	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

	# create subfolder
	model_type = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "subj{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
	folder_name = model_type.format(
		args.subject_number, 
		args.language, 
		args.num_layers, 
		args.model_type, 
		args.which_layer, 
		args.agg_type
	)
	print(folder_name)

	return folder_name

def concatenate_ranks(args):
	# "-" + str(args.batch_num) + "of" + str(args.total_batches) + "-subbatch" + str(args.sub_batch_num) + ".p"
	folder_name = find_files(args)
	get_path =	"/n/shieber_lab/Lab/users/cjou/rankings/batch-rankings-" + folder_name
	save_path =	"/n/shieber_lab/Lab/users/cjou/final_rankings/concatenated-batch-rankings-" + folder_name + ".p"
	all_rankings = []

	for i in range(args.total_batches):
		for j in range(args.total_sub_batches):
			file = get_path + "-" + str(i) + "of" + str(args.total_batches) + "-subbatch" + str(j) + ".p"
			contents = pickle.load(open(file, "rb"))
			all_rankings.extend(contents)

			del contents

	pickle.dump(all_rankings, open(save_path, "wb"))
	return

def main():
	parser = argparse.ArgumentParser("concatenate sub batch from rankings")
	parser.add_argument("-total_batches", "--total_batches", help="Total number of batches to run", type=int, default=100)
	parser.add_argument("-total_sub_batches", "--total_sub_batches", type=int, help="total number of sub_batches to run euclidean distance", required=True)
	parser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	parser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	parser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	parser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	parser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	parser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation", action='store_true', default=False)
	parser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model", action='store_true', default=False)
	parser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain", action='store_true', default=False)
	parser.add_argument("-random",  "--random", action='store_true', default=False, help="True if initialize random brain activations, False if not")
	parser.add_argument("-rand_embed",  "--rand_embed", action='store_true', default=False, help="True if initialize random embeddings, False if not")
	parser.add_argument("-glove", "--glove", action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	parser.add_argument("-word2vec", "--word2vec", action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	parser.add_argument("-bert", "--bert", action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	parser.add_argument("-permutation", "--permutation", action='store_true', default=False, help="True if permutation, False if not")
	parser.add_argument("-permutation_region", "--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	parser.add_argument("-local", "--local", action='store_true', default=False, help="True if running locally, False if not")
	args = parser.parse_args()

	concatenate_ranks(args)
	print("done.")

if __name__ == "__main__":
	main()