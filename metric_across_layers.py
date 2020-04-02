from tqdm import tqdm
import scipy.io
import pickle
import numpy as np
import sys
import argparse
import os
import random
import math
import helper
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')
# %matplotlib inline

def generate_file_name(args, subject_number, which_layer):
	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

	if args.bert or args.word2vec or args.glove:
		specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(
			bertlabel) + str(direction) + str(validate) + "-subj{}-{}_layer{}"
		file_name = specific_file.format(
			subject_number,
			args.agg_type,
			which_layer
		)
	else:
		specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(
			bertlabel) + str(direction) + str(
			validate) + "-subj{}-parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
		file_name = specific_file.format(
			subject_number,
			args.language,
			args.num_layers,
			args.model_type,
			which_layer,
			args.agg_type
		)
	return file_name

def convert_matlab_to_np(metric, volmask):
	i,j,k = volmask.shape
	nonzero_pts = np.transpose(np.nonzero(volmask))
	values = []
	for pt in tqdm(range(len(nonzero_pts))):
		x,y,z = nonzero_pts[pt]
		values.append(metric[int(x)][int(y)][int(z)])
	return values

def plot_roi_across_layers(df, metric, file_name):
	sns.set(style="darkgrid")
	plt.figure(figsize=(16, 9))
	g = sns.lineplot(x="layer", y=metric, hue="ROI", data=df)
	figure = g.get_figure()  
	box = g.get_position()
	g.set_position([box.x0, box.y0, box.width * .75, box.height])
	g.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
	figure.savefig(file_name, bbox_inches='tight')

def plot_atlas_across_layers(df, metric, file_name):
	sns.set(style="darkgrid")
	plt.figure(figsize=(24, 9))
	g = sns.lineplot(x="layer", y=metric, hue="atlas", data=df)
	figure = g.get_figure()  
	box = g.get_position()
	g.set_position([box.x0, box.y0, box.width * .85, box.height])
	g.legend(loc='center right', bbox_to_anchor=(1.6, 0.5), ncol=4)
	figure.savefig(file_name, bbox_inches='tight')

def main():
	argparser = argparse.ArgumentParser(description="layer and subject group level comparison")
	argparser.add_argument("-subject_number", "--subject_number", type=int, default=1,
						   help="subject number (fMRI data) for decoding")

	### SPECIFY MODEL PARAMETERS ###
	argparser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation",
						   action='store_true', default=False)
	argparser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model",
						   action='store_true', default=False)
	argparser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain",
						   action='store_true', default=False)
	argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str,
						   default='avg')
	argparser.add_argument("-language", "--language",
						   help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str,
						   default='spanish')
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, required=True)
	argparser.add_argument("-random", "--random", action='store_true', default=False,
						   help="True if initialize random brain activations, False if not")
	argparser.add_argument("-rand_embed", "--rand_embed", action='store_true', default=False,
						   help="True if initialize random embeddings, False if not")
	argparser.add_argument("-glove", "--glove", action='store_true', default=False,
						   help="True if initialize glove embeddings, False if not")
	argparser.add_argument("-word2vec", "--word2vec", action='store_true', default=False,
						   help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("-bert", "--bert", action='store_true', default=False,
						   help="True if initialize bert embeddings, False if not")
	argparser.add_argument("-normalize", "--normalize", action='store_true', default=False,
						   help="True if add normalization across voxels, False if not")
	argparser.add_argument("-permutation", "--permutation", action='store_true', default=False,
						   help="True if permutation, False if not")
	argparser.add_argument("-permutation_region", "--permutation_region", action='store_true', default=False,
						   help="True if permutation by brain region, False if not")
	
	### SPECIFY FOR ONE LAYER OR DIFFERENCE IN LAYERS ###
	argparser.add_argument("-across_layer", "--across_layer", help="if across layer depth significance",
						   action='store_true', default=False)

	### SPECIFY WHICH METRIC ### 
	argparser.add_argument("-fdr", "--fdr", help="if apply FDR", action='store_true', default=False)
	argparser.add_argument("-llh", "--llh", action='store_true', default=False,
						   help="True if calculate likelihood, False if not")
	argparser.add_argument("-ranking", "--ranking", action='store_true', default=False,
						   help="True if calculate ranking, False if not")
	argparser.add_argument("-rmse", "--rmse", action='store_true', default=False,
						   help="True if calculate rmse, False if not")
	argparser.add_argument("-rsa", "--rsa", action='store_true', default=False,
						   help="True if calculate rsa, False if not")

	argparser.add_argument("-local",  "--local", action='store_true', default=False, help="True if running locally")

	args = argparser.parse_args()

	if args.num_layers != 12 and args.bert:
		print("error: please ensure bert has 12 layers")
		exit()

	if args.num_layers != 1 and (args.word2vec or args.random or args.permutation or args.glove):
		print("error: please ensure baseline has 1 layer")
		exit()

	if not args.fdr and not args.llh and not args.ranking and not args.rmse:
		print("error: select at least 1 metric of correlation")
		exit()

	print("NUMBER OF LAYERS: " + str(args.num_layers))
	subjects = [1,2,4,5,7,8,9,10,11]

	# get subject 
	if args.local:
		volmask = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/volmask.p", "rb" ) )
		atlas_vals = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/atlas_vals.p", "rb" ) )
		atlas_labels = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/atlas_labels.p", "rb" ) )
		roi_vals = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/roi_vals.p", "rb" ) )
		roi_labels = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/roi_labels.p", "rb" ) )
	else:
		volmask = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/volmask.p", "rb" ) )
		atlas_vals = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/atlas_vals.p", "rb" ) )
		atlas_labels = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/atlas_labels.p", "rb" ) )
		roi_vals = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/roi_vals.p", "rb" ) )
		roi_labels = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/roi_labels.p", "rb" ) )

	num_layers = 12

	# clean labels
	final_roi_labels = helper.clean_roi(roi_vals, roi_labels)
	final_atlas_labels = helper.clean_roi(atlas_vals, atlas_labels)
	layer_info = []
	metric_info = []
	roi_info = final_roi_labels * num_layers
	atlas_info = final_atlas_labels * num_layers

	print("ROI INFO: " + str(len(roi_info)))
	print("ATLAS INFO: " + str(len(atlas_info)))

# for layer in tqdm(range(1, num_layers+1)):
# 	file_name = "bertmodel2brain_cv_-subj1-avg_layer" + str(layer)
# 	values = pickle.load(open("../final_rankings/" + str(file_name) + ".p", "rb"))
# 	metric_info.extend(values)
# 	layer_vals = len(values) * [layer]
# 	layer_info.extend(layer_vals)

	# get information
	print("getting metric information per layer...")
	for layer in tqdm(range(1, num_layers+1)):
		# file_name = generate_file_name(args, args.subject_number, layer)
		if args.local:
			# values = pickle.load(open("../final_rankings/" + str(file_name) + ".p", "rb"))
			# content = scipy.io.loadmat("../final_rankings/layer" + str(layer) + "_ranking_backwards_nifti.mat")
			# values = pickle.load(open("../final_rankings/layer" + str(layer) + "_ranking_backwards_nifti.p", "rb"))
			if args.ranking:
				values = pickle.load(open("../mat/bertmodel2brain_cv_-subj1-avg_layer" + str(layer) + "-ranking.p", "rb"))
			if args.rmse:
				# bertbrain2model_cv_-subj1-avg_layer1-3dtransform-rmse.mat
				content = scipy.io.loadmat("../mat/bertmodel2brain_cv_-subj1-avg_layer" + str(layer) + "-3dtransform-rmse.mat")["metric"]
				values = convert_matlab_to_np(content, volmask)
			if args.llh:
				content = scipy.io.loadmat("../mat/bertmodel2brain_cv_-subj1-avg_layer" + str(layer) + "-3dtransform-llh.mat")["metric"]
				values = convert_matlab_to_np(content, volmask)
		else:
			values = pickle.load(open("/n/shieber_lab/Lab/users/cjou/final_rankings/" + str(file_name) + ".p", "rb"))
		metric_info.extend(values)
		layer_vals = len(values) * [layer]
		layer_info.extend(layer_vals)

	print("LAYER INFO: " + str(len(layer_info)))
	print("METRIC INFO: " + str(len(metric_info)))

	if args.ranking:
		df_dict = {
			'layer': layer_info,
			'AR': metric_info,
			'ROI': roi_info,
			'atlas': atlas_info
		}

		df = pd.DataFrame(df_dict)

		print("plotting values...")
		plot_roi_across_layers(df, "AR", "../roi_ar_run2.png")
		plot_atlas_across_layers(df, "AR", "../atlas_ar_run2.png")

	if args.rmse:
		df_dict = {
			'layer': layer_info,
			'RMSE': metric_info,
			'ROI': roi_info,
			'atlas': atlas_info
		}

		df = pd.DataFrame(df_dict)

		print("plotting values...")
		plot_roi_across_layers(df, "RMSE", "../roi_rmse_run2.png")
		plot_atlas_across_layers(df, "RMSE", "../atlas_rmse_run2.png")

	if args.llh:
		df_dict = {
			'layer': layer_info,
			'LLH': metric_info,
			'ROI': roi_info,
			'atlas': atlas_info
		}

		df = pd.DataFrame(df_dict)

		print("plotting values...")
		plot_roi_across_layers(df, "LLH", "../roi_llh_run2.png")
		plot_atlas_across_layers(df, "LLH", "../atlas_llh_run2.png")

	print("done.")
	return

if __name__ == "__main__":
	main()