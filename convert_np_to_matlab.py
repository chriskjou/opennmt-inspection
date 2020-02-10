import numpy as np
import argparse
from tqdm import tqdm
import pickle
import scipy.io
import helper
import os
import pandas as pd
import helper

# def get_concatenated_residuals(args, file_path, file_name):
# 	if args.log:
# 		concatenated_residuals = pickle.load(open(file_path + file_name + "-transform-log-rmse.p", "rb"))
# 		return concatenated_residuals
# 	concatenated_residuals = pickle.load(open(file_path + file_name + "-transform-rmse.p", "rb"))
# 	return concatenated_residuals

# def save_to_mat(args, vals, file_name):
# 	if args.log:
# 		scipy.io.savemat("../mat/" + file_name + "-log.mat", dict(rmse = vals))
# 		print("saved file: ../mat/" + file_name + "-log.mat")
# 		return
# 	scipy.io.savemat("../mat/" + file_name + ".mat", dict(rmse = vals))
# 	print("saved file: ../mat/" + file_name + ".mat")
# 	return

def plot_average_rank_per_brain_region():
	return

def get_rankings_by_brain_region(values, atlas, roi):
	df_dict = {'voxel_index': list(range(len(values))),
		'rankings': values,
		'atlas_labels': atlas,
		'roi_labels': roi}

	df = pd.DataFrame(df_dict)
	return

def main():
	argparser = argparse.ArgumentParser(description="calculate rankings for model-to-brain")
	argparser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	argparser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	argparser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	argparser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
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
	argparser.add_argument("-normalize", "--normalize",  action='store_true', default=False, help="True if add normalization across voxels, False if not")
	argparser.add_argument("-local", "--local",  action='store_true', default=False, help="True if local, False if not")
	argparser.add_argument("-log", "--log",  action='store_true', default=False, help="True if use log coordinates, False if not")
	argparser.add_argument("-rmse", "--rmse",  action='store_true', default=False, help="True if rmse, False if not")
	argparser.add_argument("-ranking", "--ranking",  action='store_true', default=False, help="True if ranking, False if not")
	argparser.add_argument("-fdr", "--fdr",  action='store_true', default=False, help="True if fdr, False if not")
	argparser.add_argument("-llh", "--llh",  action='store_true', default=False, help="True if llh, False if not")
	args = argparser.parse_args()

	# check conditions // can remove when making pipeline
	if args.brain_to_model and args.model_to_brain:
		print("select only one flag for brain_to_model or model_to_brain")
		exit()
	if not args.brain_to_model and not args.model_to_brain:
		print("select at least flag for brain_to_model or model_to_brain")
		exit()
	if not args.rmse and not args.ranking and not args.fdr and not args.llh:
		print("select at least flag for rmse, ranking, fdr, llh")
		exit()

	print("getting volmask...")
	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)
	if args.local:
		volmask = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/volmask.p", "rb" ) )
		if args.ranking:
			atlas_vals = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/atlas_vals.p", "rb" ) )
			atlas_labels = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/atlas_labels.p", "rb" ) )
			roi_vals = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/roi_vals.p", "rb" ) )
			roi_labels = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/roi_labels.p", "rb" ) )

	else:
		volmask = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj" + str(args.subject_number) + "/volmask.p", "rb" ) )
		if args.ranking:
			atlas_vals = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/atlas_vals.p", "rb" ) )
			atlas_labels = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/atlas_labels.p", "rb" ) )
			roi_vals = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/roi_vals.p", "rb" ) )
			roi_labels = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/roi_labels.p", "rb" ) )
	
	
	### MAKE PATHS ###
	print("making paths...")
	if not os.path.exists('../mat/'):
		os.makedirs('../mat/')

	### PREDICTIONS BELOW ###
	# if args.log:
	# 	file_path = "../3d-brain-log/"
	# else:
	# 	file_path = "../3d-brain/"

	specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"	
	file_name = specific_file.format(
		args.subject_number, 
		args.language, 
		args.num_layers, 
		args.model_type, 
		args.which_layer, 
		args.agg_type
	)
	print("transform coordinates...")
	# vals = get_concatenated_residuals(args, "../rmses/concatenated-", file_name)
	if args.rmse:
		vals = pickle.load( open( "../rmses/concatenated-" + file_name + ".p", "rb" ) )
		rmses_3d = helper.transform_coordinates(vals, volmask, save_path="../mat/" + file_namec, metric="rmse")
	if args.ranking:
		vals = pickle.load( open( "../final_rankings/" + file_name + ".p", "rb" ) )
		final_roi_labels = helper.clean_roi(roi_vals, roi_labels)
		final_atlas_labels = helper.clean_atlas(atlas_vals, atlas_labels)
		get_rankings_by_brain_region(vals, final_atlas_labels, final_roi_labels)
		rankings_3d = helper.transform_coordinates(vals, volmask, save_path="../mat/" + file_name, metric="ranking")
	# print("saving matlab file...")
	# save_to_mat(args, rmses_3d, file_name)
	print('done.')

if __name__ == "__main__":
	main()