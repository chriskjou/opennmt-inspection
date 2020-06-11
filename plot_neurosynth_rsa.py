import pickle
import argparse
import pandas as pd 
from tqdm import tqdm
import numpy as np 
import helper
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')
# %matplotlib inline

def main():
	parser = argparse.ArgumentParser("plot bert neurosynth rsa")
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers", type=int, default=12)
	parser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	parser.add_argument("-local",  "--local", action='store_true', default=True, help="True if running locally")
	args = parser.parse_args()

	corrs = []
	layer_info = []
	region_info = []
	num_regions = 8

	# get correlations
	print("getting files...")
	for layer in tqdm(range(1, args.num_layers+1)):
		file_name = "bert_avg_layer{}_subj{}".format(layer, args.subject_number)
		loc = "../rsa_neurosynth/" + file_name + ".p"

		file_contents = pickle.load(open(loc, "rb"))
		corrs.append(file_contents)
		layer_vals = num_regions * [layer]
		layer_info.extend(layer_vals)
		region_info.extend(list(range(num_regions)))

	corrs = np.array(corrs)
	print(corrs.shape)

	corr_info = corrs.flatten()

	# get labels
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

	roi_info = true_roi_labels * args.num_layers
	atlas_info = true_atlas_labels * args.num_layers

	print("LAYER INFO: " + str(len(layer_info)))
	print("CORR INFO: " + str(len(corr_info)))
	print("REGION INFO: " + str(len(region_info)))
	# print("ROI INFO: " + str(len(roi_info)))
	# print("ATLAS INFO: " + str(len(atlas_info)))

	# make dictionary
	df_dict = {
		'layer': layer_info,
		'corr': corr_info,
		# 'corr': np.arctanh(corr_info),
		'region': region_info
	}

	df = pd.DataFrame(df_dict)
	sns.set(style="darkgrid")
	plt.figure(figsize=(24, 9))
	g = sns.lineplot(data=df, x='layer', y='corr', hue='region', palette=sns.color_palette('coolwarm', n_colors=201), legend="full")
	figure = g.get_figure()  
	box = g.get_position()
	g.set_position([box.x0, box.y0, box.width * .6, box.height])
	g.legend(loc='center right', bbox_to_anchor=(1.6, 0.5), ncol=6)
	# g._legend.legendHandles[0].set_alpha(.5)
	plt.savefig("../ev_neurosynth_rsa_bert_subj" + str(args.subject_number) + ".png")

	print(df.head())

	# plot 
	# helper.plot_region_across_layers(df, 'corr', "../test_neurosynth_rsa_bert_subj" + str(args.subject_number) + ".png")

	print("done.")

if __name__ == "__main__":
    main()