import numpy as np
import pickle
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import argparse
import os
from tqdm import tqdm
import math

def clean_roi(roi_vals, roi_labels):
	roi_vals = roi_vals.reshape((len(roi_vals), ))
	final_roi_labels = []
	for val_index in roi_vals:
		if val_index == 0:
			final_roi_labels.append("other")
		else:
			final_roi_labels.append(roi_labels[val_index-1][0][0])
	return final_roi_labels

def clean_atlas(atlas_vals, atlas_labels):
	at_vals = atlas_vals.reshape((len(atlas_vals), ))
	at_labels = []
	for val_index in at_vals:
		at_labels.append(atlas_labels[val_index-1][0][0])
	return at_labels

# def get_location(df, atype, layer_num, names, activations):
#     df_agg = df[df.agg_type == atype][df.layer == layer_num]
#     indices = []
#     for name in names:
#         index = df_agg.index[df_agg['atlas_labels'] == name].tolist()
#         indices += index
#     all_activations = [activations[x] for x in indices]
#     return np.nansum(all_activations), np.nansum(all_activations) // len(all_activations)
#     print("SUM: ", np.nansum(all_activations))
#     print("AVG: ", np.nansum(all_activations) // len(all_activations))

# def compare_aggregations(df):
# 	# g = sns.catplot(x="roi_labels", y="residuals", data=df, hue="agg_type", kind="bar", height=7.5, aspect=1.5)
# 	# g.set_xticklabels(rotation=90)
# 	#plt.show()
# 	return

# def plot_aggregations(df, args, file_name):
# 	all_residuals = list(df.residuals)
# 	g = sns.catplot(x="roi_labels", y="residuals", data=df, hue="layer", kind="bar", height=7.5, aspect=1.5)
# 	g.set_axis_labels("", "RMSE")
# 	g.set(ylim=(min(all_residuals), max(all_residuals)/1.75))
# 	plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language] + ", " + str(bm) + " " + str(cv))
# 	plt.show()
# 	return

def plot_atlas(df, args, file_name, zoom=False):
	all_activations = list(df.activations)
	g = sns.catplot(x="atlas_labels", y="activations", data=df, height=17.5, aspect=1.5)
	g.set_xticklabels(rotation=90)
	if zoom:
		g.set(ylim=(min(all_activations), max(all_activations) / math.pow(10, 12))) #5 * math.pow(10, -11)))
		file_name += "-zoom"
	else:
		g.set(ylim=(min(all_activations), max(all_activations)))
	g.set_axis_labels("RMSE", "")
	plt.title("Initial Activations for Subject " + str(args.subject_number) + " in all Brain Regions")
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	# plt.show()
	return

def plot_roi(df, args, file_name, zoom=False):
	all_activations = list(df.activations)
	g = sns.catplot(x="roi_labels", y="activations", data=df, height=7.5, aspect=1.5)
	g.set_xticklabels(rotation=90)
	if zoom:
		g.set(ylim=(min(all_residuals), max(all_residuals) / math.pow(10, 12))) #5 * math.pow(10, -11)))
		file_name += "-zoom"
	else:
		g.set(ylim=(min(all_activations), max(all_activations)))
	g.set_axis_labels("RMSE", "")
	plt.title("Initial Activations for Subject " + str(args.subject_number) + " in all Language Regions")
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	return

def plot_boxplot_for_atlas(df, args, file_name):
	all_activations = list(df.activations)
	g = sns.catplot(x="atlas_labels", y="activations", data=df, height=17.5, aspect=1.5, kind="box")
	g.set_xticklabels(rotation=90)
	g.set(ylim=(min(all_activations), max(all_activations)))
	g.set_axis_labels("activations", "")
	plt.title("Initial Activations for Subject " + str(args.subject_number) + " in all Brain Regions")
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	return

def plot_boxplot_for_roi(df, args, file_name):
	all_activations = list(df.activations)
	g = sns.catplot(x="roi_labels", y="activations", data=df, height=7.5, aspect=1.5, kind="box")
	g.set_xticklabels(rotation=90)
	g.set(ylim=(min(all_activations), max(all_activations)))
	g.set_axis_labels("activations", "")
	plt.title("Initial Activations for Subject " + str(args.subject_number) + " in all Language Regions")
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	return

def create_per_brain_region(activations, args, at_labels, final_roi_labels):
	# runtime error of empty mean slice
	avg = np.nanmean(activations, axis=0)

	df_dict = {'voxel_index': list(range(len(avg))),
		'activations': avg,
		'atlas_labels': at_labels,
		'roi_labels': final_roi_labels}

	df = pd.DataFrame(df_dict)

	# create plots
	print("creating plots over averaged sentence...")
	file_name = "../visualizations/initial-activations-avg-sentence-subj" + str(args.subject_number)

	plot_roi(df, args, file_name + "-roi")
	plot_atlas(df, args, file_name + "-atlas")
	plot_boxplot_for_roi(df, args, file_name + "-boxplot-roi")
	plot_boxplot_for_atlas(df, args, file_name + "-boxplot-atlas")
	return avg

def create_per_sentence(activations, args, at_labels, final_roi_labels):
	for i in tqdm(range(len(activations))):
		df_dict = {
			'voxel_index': list(range(len(activations[i]))),
			'activations': activations[i],
			'atlas_labels': at_labels,
			'roi_labels': final_roi_labels
		}

		df = pd.DataFrame(df_dict)
		to_plot = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
		# print("sentence " + str(i) + ": " + str(len(to_plot)) + " activations")

		# create plots
		print("creating plots over averaged sentence...")
		file_name = "../visualizations/initial-activations-subj" + str(args.subject_number) + "-sentence" + str(i)

		plot_roi(to_plot, args, file_name + "-roi")
		plot_atlas(to_plot, args, file_name + "-atlas")
		plot_boxplot_for_roi(to_plot, args, file_name + "-boxplot-roi")
		plot_boxplot_for_atlas(to_plot, args, file_name + "-boxplot-atlas")

	return

def main():

	argparser = argparse.ArgumentParser(description="plot initial activations by location")
	# argparser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	# argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	# argparser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	# argparser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	# argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	argparser.add_argument("-subject_number", "--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	# argparser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation", action='store_true', default=False)
	# argparser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model", action='store_true', default=False)
	# argparser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain", action='store_true', default=False)
	# argparser.add_argument("-random",  "--random", action='store_true', default=False, help="True if add cross validation, False if not")
	args = argparser.parse_args()

	# get residuals
	# check conditions // can remove when making pipeline
	# if args.brain_to_model and args.model_to_brain:
	# 	print("select only one flag for brain_to_model or model_to_brain")
	# 	exit()
	# if not args.brain_to_model and not args.model_to_brain:
	# 	print("select at least flag for brain_to_model or model_to_brain")
	# 	exit()

	# if args.brain_to_model:
	# 	direction = "brain2model_"
	# else:
	# 	direction = "model2brain_"

	# if args.cross_validation:
	# 	validate = "cv_"
	# else:
	# 	validate = "nocv_"
	# if args.random:
	# 	rlabel = "random"
	# else:
	# 	rlabel = ""


	# residual_file = sys.argv[1]
	activations = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{args.subject_number}/activations.p", "rb" ) )

	# get atlas and roi
	atlas_vals = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{args.subject_number}/atlas_vals.p", "rb" ) )
	atlas_labels = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{args.subject_number}/atlas_labels.p", "rb" ) )
	roi_vals = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{args.subject_number}/roi_vals.p", "rb" ) )
	roi_labels = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{args.subject_number}/roi_labels.p", "rb" ) )

	print("INITIAL:")
	print(len(atlas_vals))
	print(len(atlas_labels))
	print(len(roi_vals))
	print(len(roi_labels))

	final_roi_labels = clean_roi(roi_vals, roi_labels)
	at_labels = clean_atlas(atlas_vals, atlas_labels)

	print("CLEANING")
	print(len(final_roi_labels))
	print(len(at_labels))

	if not os.path.exists('../visualizations/'):
		os.makedirs('../visualizations/')

	# make dataframe
	print(len(list(range(len(activations)))))
	print(len(activations))
	print(len(at_labels))
	print(len(final_roi_labels))

	# create_per_brain_region(activations, args, at_labels, final_roi_labels)
	create_per_sentence(activations, args, at_labels, final_roi_labels)

	print("done.")

	return

if __name__ == "__main__":
    main()
