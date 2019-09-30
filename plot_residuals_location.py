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

map_dict = {
	'avg': "Average",
	'min': "Minimum", 
	'max': "Maximum",
	'last': "Last",
	"spanish": "Spanish",
	"swedish": "Swedish",
	"french": "French",
	"german": "German",
	"italian": "Italian"
}

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

def get_location(df, atype, layer_num, names, activations):
    df_agg = df[df.agg_type == atype][df.layer == layer_num]
    indices = []
    for name in names:
        index = df_agg.index[df_agg['atlas_labels'] == name].tolist()
        indices += index
    all_activations = [activations[x] for x in indices]
    return np.nansum(all_activations), np.nansum(all_activations) // len(all_activations)
    print("SUM: ", np.nansum(all_activations))
    print("AVG: ", np.nansum(all_activations) // len(all_activations))

def compare_aggregations(df):
	g = sns.catplot(x="roi_labels", y="residuals", data=df, hue="agg_type", kind="bar", height=7.5, aspect=1.5)
	# g.set_xticklabels(rotation=90)
	plt.show()
	return

def plot_aggregations(df, args, file_name):
	all_residuals = list(df.residuals)
	g = sns.catplot(x="roi_labels", y="residuals", data=df, hue="layer", kind="bar", height=7.5, aspect=1.5)
	g.set_axis_labels("", "RMSE")
	g.set(ylim=(min(all_residuals), max(all_residuals)/1.75))
	plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language])
	plt.show()
	return

def plot_atlas(df, args, file_name):
	all_residuals = list(df.residuals)
	g = sns.catplot(x="residuals", y="atlas_labels", data=df, height=17.5, aspect=1.5)
	g.set_xticklabels(rotation=90)
	g.set(xlim=(min(all_residuals), max(all_residuals)))
	g.set_axis_labels("RMSE", "location")
	plt.title("RMSE in all Brain Regions for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language])
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	# plt.show()
	return

def plot_roi(df, args, file_name):
	all_residuals = list(df.residuals)
	g = sns.catplot(x="residuals", y="roi_labels", data=df, height=7.5, aspect=1.5)
	g.set_xticklabels(rotation=90)
	g.set(xlim=(min(all_residuals), max(all_residuals)))
	g.set_axis_labels("RMSE", "location")
	plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language])
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	return

def main():
	# if len(sys.argv) != 2:
	# 	print("usage: python plot_residuals_locations.py -residual")
	# 	# example: python plot_residuals_locations.py ../residuals/concatenated_all_residuals.p
	# 	exit()

	argparser = argparse.ArgumentParser(description="plot RMSE by location")
	argparser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	argparser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	argparser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	argparser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation", action='store_true', default=False)
	argparser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model", action='store_true', default=False)
	argparser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain", action='store_true', default=False)
	argparser.add_argument("-subject_number", "--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	args = argparser.parse_args()

	# get residuals
	# check conditions // can remove when making pipeline
	if args.brain_to_model and args.model_to_brain:
		print("select only one flag for brain_to_model or model_to_brain")
		exit()
	if not args.brain_to_model and not args.model_to_brain:
		print("select at least flag for brain_to_model or model_to_brain")
		exit()

	if args.brain_to_model:
		direction = "brain2model_"
	else:
		direction = "model2brain_"

	if args.cross_validation:
		validate = "cv_"
	else:
		validate = "nocv_"

	# residual_file = sys.argv[1]
	file_loc = str(direction) + str(validate) + "subj{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
	
	file_name = file_loc.format(
		args.subject_number, 
		args.language, 
		args.num_layers, 
		args.model_type, 
		args.which_layer, 
		args.agg_type
	)

	residual_file = "../rmses/concatenated-" + str(file_name) + ".p"

	# file_name = residual_file.split("/")[-1].split(".")[0]
	all_residuals = pickle.load( open( residual_file, "rb" ) )

	# get atlas and roi
	atlas_vals = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{args.subject_number}/atlas_vals.p", "rb" ) )
	atlas_labels = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{args.subject_number}/atlas_labels.p", "rb" ) )
	roi_vals = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{args.subject_number}/roi_vals.p", "rb" ) )
	roi_labels = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{args.subject_number}/roi_labels.p", "rb" ) )

	final_roi_labels = clean_roi(roi_vals, roi_labels)
	at_labels = clean_atlas(atlas_vals, atlas_labels)

	if not os.path.exists('../visualizations/'):
		os.makedirs('../visualizations/')

	# make dataframe
	df_dict = {'voxel_index': list(range(len(all_residuals))),
			'residuals': all_residuals,
			'atlas_labels': at_labels,
			'roi_labels': final_roi_labels}

	df = pd.DataFrame(df_dict)

	plot_roi(df, args, file_name + "-roi")
	plot_atlas(df, args, file_name + "-atlas")
	# plot_aggregations(df, args, file_name + "-agg")

	print("done.")

	return

if __name__ == "__main__":
    main()
