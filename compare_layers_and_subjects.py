from tqdm import tqdm
import scipy.io
import pickle
import numpy as np
import sys
from sklearn.model_selection import KFold
import argparse
import os
from scipy.stats import pearsonr
from scipy import stats
from scipy.linalg import lstsq
import random
import math
import statsmodels.stats.multitest as smm
import matplotlib.pyplot as plt
import helper
from scipy import stats
import pandas as pd
# plt.switch_backend('agg')
import seaborn as sns
# %matplotlib inline

def calculate_pval(layer1, layer2):
	pval = stats.ttest_rel(layer1, layer2)
	return pval

def calculate_ttest(values):
	_, prob = stats.ttest_1samp(values, 0.0)
	return prob

def compare_layers(layer1, layer2):
	diff = layer1-layer2
	return diff

def get_file(args, file_name):
	if args.ranking:
		metric = "ranking"
		path = "../mat/"
	elif args.rmse:
		metric = "rmse"
		path = "../3d-brain/"
	elif args.llh:
		metric = "llh"
		path = ""
	elif args.fdr:
		metric = "fdr"
		path = "../fdr/"
	else:
		print("error: check for valid method of correlation")
	save_path = path + str(file_name) + "-3dtransform-" + str(metric)

	print("LOADING FILE: " + str(save_path) + ".mat")
	values = scipy.io.loadmat(save_path + ".mat")

	if args.fdr:
		pvals = scipy.io.loadmat(save_path + "-pvals.mat")
	else:
		pvals = []
	return values["metric"], pvals

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

def find_common_brain_space(args, subjects, which_layer):
	first = True
	for subj in subjects:
		layer_file_name = generate_file_name(args, which_layer)
		layer, _ = get_file(args, layer_file_name)
		if first:
			common_space = np.ones((layer.shape[0], layer.shape[1], layer.shape[2]))
			first = False
		curr_mask = layer > 0
		common_space = common_space & curr_mask
	return common_space


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

	### PLOTTING ###
	argparser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]",
						   type=int, default=1)

	### SPECIFY FOR SINGLE SUBJECT OR GROUP LEVEL ANALYSIS ###
	argparser.add_argument("-single_subject", "--single_subject", help="if single subject analysis",
						   action='store_true', default=False)
	argparser.add_argument("-group_level", "--group_level", help="if group level analysis", action='store_true',
						   default=False)
	argparser.add_argument("-searchlight", "--searchlight", help="if searchlight", action='store_true', default=False)
	
	### SPECIFY FOR ONE LAYER OR DIFFERENCE IN LAYERS ###
	argparser.add_argument("-single_layer", "--single_layer", help="if single layer significance",
						   action='store_true', default=False)
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

	args = argparser.parse_args()

	if args.layer1 == args.layer2:
		print("error: please select different layers for layer1 and layer2")
		exit()

	if args.num_layers != 12 and args.bert:
		print("error: please ensure bert has 12 layers")
		exit()

	if args.num_layers != 12 and (args.word2vec or args.random or args.permutation or args.glove):
		print("error: please ensure baseline has 1 layer")
		exit()

	print("NUMBER OF LAYERS: " + str(args.num_layers))
	subjects = [1,2,4,5,7,8,9,10,11]
	# print("generating file names...")
	# layer1_file_name = generate_file_name(args, args.layer1)
	# layer2_file_name = generate_file_name(args, args.layer2)

	# print("retrieving file contents...")
	# layer1 = get_file(args, layer1_file_name)
	# layer2 = get_file(args, layer2_file_name)

	# print("evaluating layers...")
	# diff = compare_layers(layer1, layer2)
	# print("DIFF")
	# print(np.sum(diff))

	# generate heatmap
	if args.single_subject and args.across_layer:
		heatmap_differences = np.zeros((args.num_layers, args.num_layers))
		for l1 in list(range(1, args.num_layers + 1)):
			for l2 in list(range(l1, args.num_layers + 1)):
				print("generating file names...")
				layer1_file_name = generate_file_name(args, args.subject_number, l1)
				layer2_file_name = generate_file_name(args, args.subject_number, l2)

				print("retrieving file contents...")
				layer1, pvals = get_file(args, layer1_file_name)
				layer2, pvals = get_file(args, layer2_file_name)

				diff = compare_layers(layer1, layer2)
				pvals = stats.ttest_rel(layer1, layer2)
				heatmap_differences[l1-1][l2-1] = np.sum(np.abs(diff))
				heatmap_differences[l2-1][l1-1] = np.sum(np.abs(diff))

		print(heatmap_differences.shape)
		print(heatmap_differences)

		Index = ['layer' + str(i) for i in range(1, args.num_layers + 1)]
		df = pd.DataFrame(heatmap_differences, index=Index, columns=Index)
		# plt.pcolor(df)
		# plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
		# plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
		# plt.show()
		sns.heatmap(df)
		plt.show()

		# pval = calculate_pval(layer1, layer2)
		# print("pval")
		# print(pval)

		# save_file(args, diff, "difference_" + a)
		# save_file(args, pval, "pval_" + )

	if args.group_level and args.across_layer:
		common_space = find_common_brain_space(args, subjects, args.which_layer)
		a,b,c = common_space.shape

		heatmap_differences = np.zeros((args.num_layers, args.num_layers))
		for l1 in list(range(1, args.num_layers + 1)):
			for l2 in list(range(l1, args.num_layers + 1)):

				# values_to_plot = np.zeros((len(subjects),a,b,c))
				layer1_vals = np.zeros((len(subjects),a,b,c))
				layer2_vals = np.zeros((len(subjects),a,b,c))

				group_avgs = []
				group_pvals = []

				for subj_index in range(len(subjects)):
					layer1_file_name = generate_file_name(args, subj_index, l1)
					layer2_file_name = generate_file_name(args, subj_index, l2)

					layer1, pvals = get_file(args, layer1_file_name)
					layer2, pvals = get_file(args, layer2_file_name)

					common_per_layer1 = layer1[common_space.astype(bool)]
					common_per_layer2 = layer2[common_space.astype(bool)]
					# pvals_per_layer = pvals[common_space.astype(bool)]

					# values_to_plot[subj_index] = common_per_layer
					# pvalues[subj_index] = pvals_per_layer
					layer1_vals[subj_index] = common_per_layer1
					layer2_vals[subj_index] = common_per_layer2

					diff = compare_layers(common_per_layer1, common_per_layer2)
					group_avgs.append(diff)
				
				pvals = stats.ttest_rel(layer1_vals, layer2_vals)
				heatmap_differences[l1-1][l2-1] = np.sum(np.abs(np.mean(avgs, axis=0)))
				heatmap_differences[l2-1][l1-1] = np.sum(np.abs(np.mean(avgs, axis=0)))

		print(heatmap_differences.shape)
		print(heatmap_differences)

		Index = ['layer' + str(i) for i in range(1, args.num_layers + 1)]
		df = pd.DataFrame(heatmap_differences, index=Index, columns=Index)
		# plt.pcolor(df)
		# plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
		# plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
		# plt.show()
		sns.heatmap(df)
		plt.show()
		pass

	if args.group_level and args.single_layer:
		common_space = find_common_brain_space(args, subjects, args.which_layer)
		a,b,c = common_space.shape
		corr = np.zeros((len(subjects),a,b,c))
		pvalues = np.zeros((len(subjects),a,b,c))

		# get common values
		for subj_index in range(len(subjects)):
			layer_file_name = generate_file_name(args, subjects[subj_index], args.which_layer)
			layer, pvals = get_file(args, args.which_layer)
			common_per_layer = layer[common_space.astype(bool)]
			pvals_per_layer = pvals[common_space.astype(bool)]
			corr[subj_index] = common_per_layer
			pvalues[subj_index] = pvals_per_layer
		

		group_pvals = np.apply_along_axis(calculate_ttest, 0, pvalues)
		group_corrs = np.mean(corr, axis=0)

		save_location = "/n/shieber_lab/Lab/users/cjou/fdr/group_level_single_layer_" + str(args.which_layer)
		volmask = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj" + str(args.subject_number) + "/volmask.p", "rb" ) )
		_ = helper.transform_coordinates(group_corrs, volmask, save_location, "fdr", pvals=group_pvals)

	print("done.")
	return

if __name__ == "__main__":
	main()