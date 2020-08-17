import numpy as np
import pickle
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.switch_backend('agg')
import argparse
import os
from tqdm import tqdm
import math
import helper
import scipy.io

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
	sns.set(style="darkgrid")
	g = sns.catplot(x="atlas_labels", y="activations", data=df, height=17.5, aspect=1.5, kind="box", color="cornflowerblue")
	g.set_xticklabels(rotation=90)
	g.set(ylim=(min(all_activations), max(all_activations)))
	g.set_axis_labels("", "activations")
	plt.title("Initial Activations for Subject " + str(args.subject_number) + " in all AAL Brain Regions")
	plt.savefig("../visualizations/run2-" + str(file_name) + ".png", bbox_inches='tight')
	return

def plot_boxplot_for_roi(df, args, file_name):
	df = df.sort_values(by=["roi_labels"])
	all_activations = list(df.activations)
	sns.set(style="darkgrid")
	g = sns.catplot(x="roi_labels", y="activations", data=df, height=7.5, aspect=1.5, kind="box", color="cornflowerblue")
	g.set_xticklabels(rotation=45)
	g.set(ylim=(min(all_activations), max(all_activations)))
	g.set_axis_labels("", "activations")
	plt.title("Initial Activations for Subject " + str(args.subject_number) + " in all Language ROIs")
	plt.savefig("../visualizations/run2-" + str(file_name) + ".png", bbox_inches='tight')
	return

def create_per_brain_region(activations, args, at_labels, final_roi_labels):
	# runtime error of empty mean slice
	avg = np.nanmean(activations, axis=0)

	df_dict = {'voxel_index': list(range(len(avg))),
		'activations': avg,
		'atlas_labels': at_labels,
		'roi_labels': final_roi_labels}

	df = pd.DataFrame(df_dict)
	labels = list(set(at_labels))
	labels.sort()
	print(labels)

	# create plots
	print("creating plots over averaged sentence...")
	file_name = "initial-activations-avg-sentence-subj" + str(args.subject_number)
	# file_name = "../visualizations/run2-initial-activations-avg-sentence-subj" + str(args.subject_number)

	# plot_roi(df, args, file_name + "-roi")
	# plot_atlas(df, args, file_name + "-atlas")
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
		# print("creating plots over averaged sentence...")
		file_name = "initial-activations-subj" + str(args.subject_number) + "-sentence" + str(i)

		# plot_roi(to_plot, args, file_name + "-roi")
		# plot_atlas(to_plot, args, file_name + "-atlas")
		plot_boxplot_for_roi(to_plot, args, file_name + "-boxplot-roi")
		plot_boxplot_for_atlas(to_plot, args, file_name + "-boxplot-atlas")

	return

def plot_voxel_num(df, metric):
	if metric == "atlas_labels":
		height = 17.5
		aspect = 1.5
		df = df[[metric]].groupby([metric]).size().reset_index(name='num_voxels').sort_values(by=['num_voxels'])
	else:
		height = 7.5
		aspect = 1.5
		df = df[[metric]].groupby([metric]).size().reset_index(name='num_voxels').sort_values(by=[metric])

	if metric == "roi_labels":
		df = df.loc[df[metric] != "other"]
	plt.clf()
	sns.set(style="darkgrid")
	g = sns.catplot(x=metric, y="num_voxels", height=height, aspect=aspect, data=df, kind="bar", color="cornflowerblue")
	g.set_xticklabels(rotation=90)
	plt.savefig("../"+ str(metric) + "_hist.png", bbox_inches='tight')
	print("TOTAL: " + str(np.sum(df['num_voxels'])))
	return

def main():

	argparser = argparse.ArgumentParser(description="plot initial activations by location")
	argparser.add_argument("-subject_number", "--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	argparser.add_argument("-subjects", "--subjects", help="subject numbers", type=str, default="")
	argparser.add_argument("-local", "--local",  action='store_true', default=False, help="True if local False if not")
	argparser.add_argument("-brain_map", "--brain_map",  action='store_true', default=False, help="True if for 3d brain map if not")
	argparser.add_argument("-hist", "--hist",  action='store_true', default=False, help="True if for histogram of voxels if not")
	argparser.add_argument("-sentences", "--sentences",  help="sentence numbers in numbers with commas", type=str, default="", required=True)
	argparser.add_argument("-aal", "--aal",  action='store_true', default=False, help="True if all brain AAL regions False if not")

	### UPDATE FILE PATHS HERE ###
	argparser.add_argument("-fmri_path", "--fmri_path", default="/n/shieber_lab/Lab/users/cjou/fmri/", type=str, help="file path to fMRI data on the Odyssey cluster")
	### UPDATE FILE PATHS HERE ###

	args = argparser.parse_args()

	if args.brain_map:
		subject_numbers = [int(subj_num) for subj_num in args.subjects.split(",")]  
		sentence_numbers = [int(subj_num) for subj_num in args.sentences.split(",")]  

		print("finding common brain space...")
		volmask = helper.load_common_space(subject_numbers, local=args.local)

		print("getting all activations...")
		activations_list = []
		for subj_num in tqdm(subject_numbers):
			print("adding subject: " + str(subj_num))
			if args.local:
				file_name = "../examplesGLM/subj" + str(subj_num) + "/modified_activations.p"
			else:
				file_name = "/n/shieber_lab/Lab/users/cjou/fmri/subj" + str(subj_num) + "/modified_activations.p"
			print("FILE NAME: " + str(file_name))
			activations = pickle.load(open(file_name, "rb"))
			if len(sentence_numbers) > 0 and subj_num == 1:
				for sent_num in sentence_numbers:
					scipy.io.savemat("../mat/subj" + str(subj_num) + "_initial_activations_sentence" + str(sent_num) + ".mat", dict(metric = np.array(activations)[sent_num-1]))
			avg_acts_per_subject = np.mean(np.array(activations), axis=0)
			scipy.io.savemat("../mat/subj" + str(subj_num) + "_initial_activations.mat", dict(metric = avg_acts_per_subject))
			common_act = np.ma.array(avg_acts_per_subject, mask=volmask)
			activations_list.append(common_act)

		print("saving average activations...")
		across_brain = np.mean(np.array(activations_list), axis=0)
		scipy.io.savemat("../mat/common_space_initial_activations.mat", dict(metric = across_brain))

	elif args.hist: 
		if args.local:
			volmask = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/volmask.p", "rb" ) )
			activations = pickle.load( open( "../examplesGLM/subj" + str(args.subject_number) + "/activations.p", "rb" ) )
			atlas_vals = pickle.load( open("../examplesGLM/subj" + str(args.subject_number) +  "/atlas_vals.p", "rb" ) )
			atlas_labels = pickle.load( open( "../examplesGLM/subj" + str(args.subject_number) + "/atlas_labels.p", "rb" ) )
			roi_vals = pickle.load( open( "../examplesGLM/subj" + str(args.subject_number) +  "/roi_vals.p", "rb" ) )
			roi_labels = pickle.load( open( "../examplesGLM/subj" + str(args.subject_number) + "/roi_labels.p", "rb" ) )

		final_roi_labels = helper.compare_labels(roi_labels, volmask, roi=True)
		final_atlas_labels = helper.compare_labels(atlas_labels, volmask)

		avg = np.nanmean(activations, axis=0)

		df_dict = {'voxel_index': list(range(len(avg))),
			'activations': avg,
			'atlas_labels': final_atlas_labels,
			'roi_labels': final_roi_labels
		}

		df = pd.DataFrame(df_dict)

		# PLOT ALTAS
		if args.aal:
			plot_voxel_num(df, "atlas_labels")
		else:
			plot_voxel_num(df, "roi_labels")
	else:
		# get atlas and roi
		if args.local:
			volmask = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/volmask.p", "rb" ) )
			activations = pickle.load( open( "../examplesGLM/subj" + str(args.subject_number) + "/activations.p", "rb" ) )
			atlas_vals = pickle.load( open("../examplesGLM/subj" + str(args.subject_number) +  "/atlas_vals.p", "rb" ) )
			atlas_labels = pickle.load( open( "../examplesGLM/subj" + str(args.subject_number) + "/atlas_labels.p", "rb" ) )
			roi_vals = pickle.load( open( "../examplesGLM/subj" + str(args.subject_number) +  "/roi_vals.p", "rb" ) )
			roi_labels = pickle.load( open( "../examplesGLM/subj" + str(args.subject_number) + "/roi_labels.p", "rb" ) )
		else:
			volmask = pickle.load( open( "{}subj{}/volmask.p".format(args.fmri_path, args.subject_number), "rb" ) )
			activations = pickle.load( open( "{}subj{}/activations.p".format(args.fmri_path, args.subject_number), "rb" ) )
			atlas_vals = pickle.load( open( "{}subj{}/atlas_vals.p".format(args.fmri_path, args.subject_number), "rb" ) )
			atlas_labels = pickle.load( open( "{}subj{}/atlas_labels.p".format(args.fmri_path, args.subject_number), "rb" ) )
			roi_vals = pickle.load( open( "{}subj{}/roi_vals.p".format(args.fmri_path, args.subject_number), "rb" ) )
			roi_labels = pickle.load( open( "{}subj{}/roi_labels.p".format(args.fmri_path, args.subject_number), "rb" ) )

		print("INITIAL:")
		print(len(atlas_vals))
		print(len(atlas_labels))
		print(len(roi_vals))
		print(len(roi_labels))

		final_roi_labels = helper.compare_labels(roi_labels, volmask, roi=True)
		final_atlas_labels = helper.compare_labels(atlas_labels, volmask)
		# final_roi_labels = clean_roi(roi_vals, roi_labels)
		# at_labels = clean_atlas(atlas_vals, atlas_labels)

		print("CLEANING")
		print(len(final_roi_labels))
		print(len(final_atlas_labels))

		if not os.path.exists('../visualizations/'):
			os.makedirs('../visualizations/')

		# make dataframe
		print(len(list(range(len(activations)))))
		print(len(activations))
		print(len(final_atlas_labels))
		print(len(final_roi_labels))

		create_per_brain_region(activations, args, final_atlas_labels, final_roi_labels)
		# create_per_sentence(activations, args, final_atlas_labels, final_roi_labels)

	print("done.")

	return

if __name__ == "__main__":
    main()
