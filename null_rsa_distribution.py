import pickle
import numpy as np 
import scipy.io
import pandas as pd
import argparse
from tqdm import tqdm
import helper
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')

def get_values_per_subject(args, subj_num):
	# get file name
	region_vals = []
	region_labels = []
	layer_labels = []

	if args.aal:
		file_name = "rsa_neurosynth_aal/"
	else:
		file_name = "rsa_neurosynth_roi/"

	for layer in tqdm(range(1, args.num_layers+1)):
		temp_file_name = "bert_avg_layer" + str(layer) + "_subj" + str(subj_num)

		# distribution
		vals = pickle.load(open("../" + str(file_name) + temp_file_name + ".p", "rb"))
		
		# get null distributions
		true_pval = np.array(pickle.load(open("../" + str(file_name) + temp_file_name + "_pval.p", "rb")))
		true_mean = np.array(pickle.load(open("../" + str(file_name) + temp_file_name + "_mean.p", "rb")))
		true_std = np.array(pickle.load(open("../" + str(file_name) + temp_file_name + "_std.p", "rb")))

		# print(temp_file_name)
		# print(np.array(vals).shape)
		# print(np.array(true_pval).shape)
		# print(np.array(true_mean).shape)
		# print(np.array(true_std).shape)

		standardized = (vals - true_mean) / true_std

		region_vals.extend(standardized)
		region_labels.extend(list(range(len(standardized))))
		layer_labels.extend([layer] * len(standardized))
	return region_vals, region_labels, layer_labels

def plot_per_subject(args):
	print("getting values...")
	region_vals, region_labels, layer_labels = get_values_per_subject(args, args.subject_number)

	print("VALS: " + str(len(region_vals)))
	print("REGIONS: " + str(len(region_labels)))
	print("LAYERS: " + str(len(layer_labels)))

	# create dataframe
	df = pd.DataFrame({
		'layer': layer_labels,
		'corr': region_vals,
		'region': region_labels
		})

	print(df.head())

	volmask = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/volmask.p", "rb" ) )
	if args.aal:
		aal_labels = pickle.load( open( "../examplesGLM/subj" + str(args.subject_number) + "/atlas_labels.p", "rb" ) )

		labels = [str(elem[0][0]) for elem in aal_labels]
		print(labels)

	else:
		roi_labels = pickle.load( open( "../examplesGLM/subj" + str(args.subject_number) + "/roi_labels.p", "rb" ) )

		labels = [str(elem[0][0]) for elem in roi_labels]
		print(labels)

	# sns.set_palette(sns.color_palette("RdBu", n_colors=201))

	# print("PALETTE: " + str(len(palette)))
	print("plotting...")
	plt.clf()
	sns.set(style="darkgrid")
	# plt.figure(figsize=(9, 9))
	g = sns.pointplot(x="layer", y="corr", hue="region", data=df, plot_kws=dict(alpha=0.3))
	figure = g.get_figure()  
	box = g.get_position()
	g.set_position([box.x0, box.y0, box.width, box.height])
	leg_handles = g.get_legend_handles_labels()[0]
	g.legend(leg_handles, labels, title="ROI", loc='center right', bbox_to_anchor=(1.35, 0.5))
	# g.legend(title="ROI", loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1, labels=labels)
	file_name = "../ev_null_rsa_distribution_subj" + str(args.subject_number) 

	if args.aal:
		file_name += "_aal"
	else:
		file_name += "_roi"

	figure.savefig(str(file_name) + ".png", bbox_inches='tight')
	return layer_vals

def plot_across_subjects(args):
	subjects = [1,2,4,5,7,8,9,10,11]
	all_region_vals = []
	all_region_labels = []
	all_layer_labels = []
	if args.aal:
		aal_labels = pickle.load( open( "../examplesGLM/subj" + str(args.subject_number) + "/atlas_labels.p", "rb" ) )
		labels = [str(elem[0][0]) for elem in aal_labels]
		print(labels)
	else:
		labels = ['LMidPostTemp', 'LPostTemp', 'LMidAntTemp', 'LIFG', 'LAntTemp', 'LIFGorb', 'LAngG', 'LMFG']

	print("getting values...")
	for subj_num in subjects:
		region_vals, region_labels, layer_labels = get_values_per_subject(args, subj_num)
		all_region_vals.extend(region_vals)
		all_region_labels.extend(region_labels)
		all_layer_labels.extend(layer_labels)

	print("ALL VALS: " + str(len(all_region_vals)))
	print("ALL REGIONS: " + str(len(all_region_labels)))
	print("ALL LAYERS: " + str(len(all_layer_labels)))

	# create dataframe
	df = pd.DataFrame({
		'layer': all_layer_labels,
		'corr': all_region_vals,
		'region': all_region_labels
		})

	print(df.head())
	
	# sns.set_palette(sns.color_palette("RdBu", n_colors=201))

	# print("PALETTE: " + str(len(palette)))
	print("plotting...")

	if args.aal:
		plt.clf()
		sns.set(style="darkgrid")
		plt.figure(figsize=(36, 9))
		g = sns.lineplot(data=df, x='layer', y='corr', hue='region', palette=sns.color_palette('coolwarm', n_colors=116), ci=68, legend="full")
		figure = g.get_figure()  
		box = g.get_position()
		g.set_position([box.x0, box.y0, box.width * .6, box.height])
		# g.legend(loc='center right', bbox_to_anchor=(1.6, 0.5), ncol=6)
		leg_handles = g.get_legend_handles_labels()[0]
		g.legend(leg_handles, labels, title="ROI", loc='center right', bbox_to_anchor=(1.6, 0.5), ncol=4)
	else:
		plt.clf()
		sns.set(style="darkgrid")
		plt.figure(figsize=(10, 9))
		g = sns.pointplot(x="layer", y="corr", hue="region", data=df, plot_kws=dict(alpha=0.5), ci=68, dodge=0.75, join=True)
		figure = g.get_figure()  
		box = g.get_position()
		g.set_position([box.x0, box.y0, box.width, box.height])
		leg_handles = g.get_legend_handles_labels()[0]
		g.legend(leg_handles, labels, title="ROI", loc='center right', bbox_to_anchor=(1.35, 0.5))

	file_name = "../ev_null_rsa_distribution_across_subjects"
	if args.aal:
		file_name += "_aal"
	else:
		file_name += "_roi"
	figure.savefig(str(file_name) + ".png", bbox_inches='tight')
	return

def main():
	argparser = argparse.ArgumentParser(description="plot rsa against null distribution of neurosynth")
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=12)
	argparser.add_argument("-subject_number", "--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	argparser.add_argument("-across_subjects",  "--across_subjects", action='store_true', default=False, help="True if running across_subjects")
	argparser.add_argument("-aal",  "--aal", action='store_true', default=False, help="True if use RSA aal null distribution")
	args = argparser.parse_args()

	if args.across_subjects:
		plot_across_subjects(args)
	else:
		plot_per_subject(args)

if __name__ == "__main__":
	main()