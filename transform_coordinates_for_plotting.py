import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm
import helper

def main():
	# parse arguments
	argparser = argparse.ArgumentParser(description="transform coordinates for plotting")
	argparser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	argparser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	argparser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	argparser.add_argument("-subject_number", "--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	argparser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation", action='store_true', default=False)
	argparser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model", action='store_true', default=False)
	argparser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain", action='store_true', default=False)
	argparser.add_argument("-glove", "--glove", action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	argparser.add_argument("-word2vec", "--word2vec", action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("-bert", "--bert", action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("-rand_embed", "--rand_embed", action='store_true', default=False, help="True if initialize random embeddings, False if not")
	argparser.add_argument("-random",  "--random", action='store_true', default=False, help="True if add cross validation, False if not")
	argparser.add_argument("-permutation",  "--permutation", action='store_true', default=False, help="True if permutation, False if not")
	argparser.add_argument("-permutation_region", "--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	
	# metrics
	argparser.add_argument("-rmse", "--rmse",  action='store_true', default=False, help="True if rmse, False if not")
	argparser.add_argument("-fdr", "--fdr",  action='store_true', default=False, help="True if fdr, False if not")
	argparser.add_argument("-rank", "--rank",  action='store_true', default=False, help="True if rank, False if not")
	
	### UPDATE FILE PATHS HERE ###
	argparser.add_argument("--fmri_path", default="/n/shieber_lab/Lab/users/cjou/fmri/", type=str, help="file path to fMRI data on the Odyssey cluster")
	argparser.add_argument("--to_save_path", default="/n/shieber_lab/Lab/users/cjou/", type=str, help="file path to and create rmse/ranking/llh on the Odyssey cluster")
	### UPDATE FILE PATHS HERE ###

	args = argparser.parse_args()

	# verify arguments
	if args.rmse and args.fdr and args.rank:
		print("select only one flag for rmse, fdr, or rank")
		exit()
	if not args.rmse and not args.fdr and not args.rank:
		print("select at least flag for rmse, fdr, or rank")
		exit()

	print("getting arguments...")
	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)
	file_loc = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "subj{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
	file_name = file_loc.format(
		args.subject_number, 
		args.language, 
		args.num_layers, 
		args.model_type, 
		args.which_layer, 
		args.agg_type
	)

	if not os.path.exists(str(args.to_save_path) + '3d-brain/'):
		os.makedirs(str(args.to_save_path) + '3d-brain/')
	save_location = str(args.to_save_path) + '3d-brain/'

	# set save location
	if args.rmse:
		# TODO
		open_location = str(args.to_save_path) + "rmse/" + str(file_name) + "_subj" + str(args.subject_number)
		metric = "rmse"
	elif args.fdr:
		open_location = str(args.to_save_path) + "fdr/" + str(file_name) + "_subj" + str(args.subject_number)
		metric = "fdr"
		points = pickle.load(open(open_location + "_valid_correlations_2d_coordinates.p", "rb"))
	elif args.rank:
		# TODO
		open_location = str(args.to_save_path) + "rankings_od32/" + str(file_name) + "_subj" + str(args.subject_number)
		metric = "rank"
	else:
		print("error")
		exit()

	# get volmask
	file_path = str(args.fmri_path) + "examplesGLM/subj{}/volmask.p".format(args.subject_number)
	volmask = pickle.load( open( file_path, "rb" ) )

	# transform coordinates and save
	print("METRIC: "+ str(metric))
	_ = helper.transform_coordinates(points, volmask, save_location=save_location, metric=metric)
	print("done.")
	
if __name__ == "__main__":
	main()