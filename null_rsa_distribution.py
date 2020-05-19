import pickle
import numpy as np 
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')

def main():
	argparser = argparse.ArgumentParser(description="plot rsa against null distribution of neurosynth")
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=12)
	argparser.add_argument("-subject_number", "--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	args = argparser.parse_args()

	# get file name
	region_vals = []
	region_labels = []
	layer_labels = []

	print("getting values...")
	for layer in tqdm(range(1, args.num_layers+1)):
		temp_file_name = "bert_avg_layer" + str(layer) + "_subj" + str(args.subject_number)

		# distribution
		vals = pickle.load(open("../rsa_neurosynth/" + temp_file_name + ".p", "rb"))
		
		# get null distributions
		true_pval = np.array(pickle.load(open("../rsa_neurosynth/" + temp_file_name + "_pval.p", "rb")))
		true_mean = np.array(pickle.load(open("../rsa_neurosynth/" + temp_file_name + "_mean.p", "rb")))
		true_std = np.array(pickle.load(open("../rsa_neurosynth/" + temp_file_name + "_mean.p", "rb")))

		standardized = (vals - true_mean) / true_std

		region_vals.extend(standardized)
		region_labels.extend(list(range(len(standardized))))
		layer_labels.extend([layer] * len(standardized))

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

	# sns.set_palette(sns.color_palette("RdBu", n_colors=201))

	# print("PALETTE: " + str(len(palette)))
	print("plotting...")
	sns.set(style="darkgrid")
	plt.figure(figsize=(24, 9))
	g = sns.pointplot(x="layer", y="corr", hue="region", data=df, plot_kws=dict(alpha=0.3))
	figure = g.get_figure()  
	box = g.get_position()
	g.set_position([box.x0, box.y0, box.width * .85, box.height])
	g.legend(loc='center right', bbox_to_anchor=(1.6, 0.5), ncol=4)
	figure.savefig("../test_null_rsa_distribution.png", bbox_inches='tight')

if __name__ == "__main__":
	main()