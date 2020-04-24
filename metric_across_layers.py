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

def plot_roi_across_layers(df, metric, file_name):
	sns.set(style="darkgrid")
	plt.figure(figsize=(16, 9))
	g = sns.pointplot(x="layer", y=metric, hue="ROI", data=df, plot_kws=dict(alpha=0.3))
	figure = g.get_figure()  
	box = g.get_position()
	g.set_position([box.x0, box.y0, box.width * .75, box.height])
	g.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
	figure.savefig(file_name, bbox_inches='tight')

def plot_atlas_across_layers(df, metric, file_name):
	sns.set(style="darkgrid")
	plt.figure(figsize=(24, 9))
	g = sns.pointplot(x="layer", y=metric, hue="atlas", data=df, plot_kws=dict(alpha=0.3))
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

	if not args.fdr and not args.llh and not args.ranking and not args.rmse and not args.rsa:
		print("error: select at least 1 metric of correlation")
		exit()

	print("NUMBER OF LAYERS: " + str(args.num_layers))
	subjects = [1,2,4,5,7,8,9,10,11]

	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

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

	true_roi_labels = helper.compare_labels(roi_labels, volmask, subj_num=args.subject_number, roi=True)
	true_atlas_labels = helper.compare_labels(atlas_labels, volmask, subj_num=args.subject_number)

	# clean labels
	final_roi_labels = helper.clean_roi(roi_vals, roi_labels)
	final_atlas_labels = helper.clean_atlas(atlas_vals, atlas_labels)
	layer_info = []
	metric_info = []
	roi_info = true_roi_labels * args.num_layers
	atlas_info = true_atlas_labels * args.num_layers

	# print("ROI INFO: " + str(len(roi_info)))
	# print("ATLAS INFO: " + str(len(atlas_info)))

	# total=0
	# print("ATLAS: " + str(len(final_atlas_labels)))
	# for i in range(len(true_atlas_labels)):
	# 	if true_atlas_labels[i] == final_atlas_labels[i]:
	# 		total+=1
	# print("TOTAL: " + str(total))

	# total=0
	# print("ROI: " + str(len(true_roi_labels)))
	# for i in range(len(true_roi_labels)):
	# 	if true_roi_labels[i] == final_roi_labels[i]:
	# 		total+=1
	# print("TOTAL: " + str(total))
	# exit()

# for layer in tqdm(range(1, num_layers+1)):
# 	file_name = "bertmodel2brain_cv_-subj1-avg_layer" + str(layer)
# 	values = pickle.load(open("../final_rankings/" + str(file_name) + ".p", "rb"))
# 	metric_info.extend(values)
# 	layer_vals = len(values) * [layer]
# 	layer_info.extend(layer_vals)

	# get information
	if args.bert:
		print("getting metric information per layer...")
		for layer in tqdm(range(1, args.num_layers+1)):
			file_name = "bert{}{}-subj{}-{}_layer{}".format(
					direction,
					validate,
					args.subject_number,
					args.agg_type,
					layer
				)
			if args.local:
				# values = pickle.load(open("../final_rankings/" + str(file_name) + ".p", "rb"))
				# content = scipy.io.loadmat("../final_rankings/layer" + str(layer) + "_ranking_backwards_nifti.mat")
				# values = pickle.load(open("../final_rankings/layer" + str(layer) + "_ranking_backwards_nifti.p", "rb"))
				if args.ranking:
					# values = pickle.load(open("../mat/bertmodel2brain_cv_-subj1-avg_layer" + str(layer) + "-ranking.p", "rb"))
					content = scipy.io.loadmat("../mat/" + str(file_name) + "-3dtransform-ranking.mat")["metric"]
				if args.rmse:
					# bertbrain2model_cv_-subj1-avg_layer1-3dtransform-rmse.mat
					content = scipy.io.loadmat("../mat/" + str(file_name) + "-3dtransform-rmse.mat")["metric"]
				if args.llh:
					content = np.abs(scipy.io.loadmat("../mat/" + str(file_name) + "-3dtransform-llh.mat")["metric"])
				if args.rsa:
					content = scipy.io.loadmat("../mat/" + str(file_name) + "-3dtransform-rsa.mat")["metric"]
				values = helper.convert_matlab_to_np(content, volmask)
			else:
				values = pickle.load(open("/n/shieber_lab/Lab/users/cjou/final_rankings/" + str(file_name) + ".p", "rb"))
			metric_info.extend(values)
			layer_vals = len(values) * [layer]
			layer_info.extend(layer_vals)
		to_save_file = str(plabel) + str(prlabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj" + str(args.subject_number) + "-bert"
	elif not args.glove and not args.word2vec:
		for layer in tqdm(range(1, args.num_layers+1)):
			file_name = "{}{}-subj{}-parallel-english-to-{}-model-{}layer-brnn-pred-layer{}-{}-3dtransform-".format(
					direction,
					validate,
					args.subject_number,
					args.language,
					args.num_layers,
					layer,
					args.agg_type
				)
			print(file_name)
			if args.local:
				if args.ranking:
					content = scipy.io.loadmat("../mat/" + file_name + "ranking.mat")["metric"]
				if args.rmse:
					content = scipy.io.loadmat("../mat/" + file_name + "rmse.mat")["metric"]
				if args.llh:
					content = np.abs(scipy.io.loadmat("../mat/" + file_name + "llh.mat")["metric"])
				
				values = helper.convert_matlab_to_np(content, volmask)
			else:
				values = pickle.load(open("/n/shieber_lab/Lab/users/cjou/final_rankings/" + str(file_name) + ".p", "rb"))
			metric_info.extend(values)
			layer_vals = len(values) * [layer]
			layer_info.extend(layer_vals)
			to_save_file = "{}_{}_subj{}_{}layer_{}".format(direction, validate, args.subject_number, args.num_layers, args.language)
		else: # word2vec, glove
			pass

	print("LAYER INFO: " + str(len(layer_info)))
	print("METRIC INFO: " + str(len(metric_info)))

	if args.ranking and args.model_to_brain:
		df_dict = {
			'layer': layer_info,
			'AR': metric_info,
			'ROI': roi_info,
			'atlas': atlas_info
		}

		df = pd.DataFrame(df_dict)

		df_slice = df.loc[df["layer"] == 1][["atlas", "AR"]]
		avg_df = df_slice.groupby(['atlas']).mean()
		print(avg_df.sort_values(by='AR', ascending=False).head())

		print("plotting values...")
		plot_roi_across_layers(df, "AR", "../fixed_roi_ar_" + to_save_file + ".png")
		plot_atlas_across_layers(df, "AR", "../fixed_atlas_ar_" + to_save_file + ".png")

	if args.rmse:
		df_dict = {
			'layer': layer_info,
			'RMSE': metric_info,
			'ROI': roi_info,
			'atlas': atlas_info
		}

		df = pd.DataFrame(df_dict)

		df_slice = df.loc[df["layer"] == 1][["atlas", "RMSE"]]
		avg_df = df_slice.groupby(['atlas']).mean()
		print(avg_df.sort_values(by='RMSE', ascending=True).head())

		print("plotting values...")
		plot_roi_across_layers(df, "RMSE", "../fixed_roi_rmse_" + to_save_file + ".png")
		plot_atlas_across_layers(df, "RMSE", "../fixed_atlas_rmse_" + to_save_file + ".png")

	if args.llh:
		df_dict = {
			'layer': layer_info,
			'LLH': metric_info,
			'ROI': roi_info,
			'atlas': atlas_info
		}

		df = pd.DataFrame(df_dict)

		df_slice = df.loc[df["layer"] == 1][["atlas", "LLH"]]
		avg_df = df_slice.groupby(['atlas']).mean()
		print(avg_df.sort_values(by='LLH', ascending=True).head())

		print("plotting values...")
		plot_roi_across_layers(df, "LLH", "../fixed_roi_llh_" + to_save_file + ".png")
		plot_atlas_across_layers(df, "LLH", "../fixed_atlas_llh_" + to_save_file + ".png")

	if args.rsa:
		df_dict = {
			'layer': layer_info,
			'correlation_coefficient': metric_info,
			'ROI': roi_info,
			'atlas': atlas_info
		}

		df = pd.DataFrame(df_dict)

		print("ATLAS...")
		df_slice = df.loc[df["layer"] == 1][["atlas", "correlation_coefficient"]]
		avg_df = df_slice.groupby(['atlas']).mean()
		print(avg_df.sort_values(by='correlation_coefficient', ascending=True).head())
		print(avg_df.sort_values(by='correlation_coefficient', ascending=False).head())

		print("ROI...")
		df_slice = df.loc[df["layer"] == 1][["ROI", "correlation_coefficient"]]
		avg_df = df_slice.groupby(['ROI']).mean()
		print(avg_df.sort_values(by='correlation_coefficient', ascending=True).head())
		print(avg_df.sort_values(by='correlation_coefficient', ascending=False).head())

		print("plotting values...")
		plot_roi_across_layers(df, "correlation_coefficient", "../fixed_roi_rsa_" + to_save_file + ".png")
		plot_atlas_across_layers(df, "correlation_coefficient", "../fixed_atlas_rsa_" + to_save_file + ".png")

	print("done.")
	return

if __name__ == "__main__":
	main()