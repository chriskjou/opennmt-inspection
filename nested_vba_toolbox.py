import pickle
import argparse
import pandas as pd 
from tqdm import tqdm
import numpy as np 
import helper
import scipy.io
from scipy.stats import linregress, ttest_1samp, ttest_ind
from statsmodels.stats.anova import AnovaRM 
import matplotlib.pyplot as plt
import seaborn as sns
import helper
plt.switch_backend('agg')

def get_counts(df):
	dims = df.shape
	vox_counts = np.zeros((dims[0], dims[1]))
	max_per_row = np.argmax(df, axis=1) 
	for elem in range(len(max_per_row)):
		which_index = max_per_row[elem]
		vox_counts[elem][which_index] = 1
	return np.sum(np.array(vox_counts), axis=0)

def get_file(args, batch_num, total_batches):
	file_name = "bor_pxp{}of{}_bor_pxp.mat".format(batch_num, total_batches)
	if args.family or args.bert:
		file_name = "family_" + file_name
	# if args.bert:
	# 	file_name = "bert_only_" + file_name
	file_contents = scipy.io.loadmat("../nested_llh_mat/" + str(file_name))["llh"]
	bor = file_contents["bor"][0][0][0]
	pxp = file_contents["pxp"][0][0]
	if args.family:
		family = file_contents["family"][0][0][0]
	else: 
		family = []
	return bor, pxp, family

def concatenate_files(args, total_batches=100):
	bors = []
	pxps = []
	families = []
	num_families = 3
	for i in tqdm(range(total_batches)):
		bor, pxp, family = get_file(args, i, total_batches)
		if args.family:
			num_voxels = family.shape[0]
			family = np.reshape(family, (num_voxels // num_families, num_families))
			families.extend(family)
		bors.extend(bor)
		pxps.extend(pxp)
	return np.array(bors), np.array(pxps), np.array(families)

def plot_count_graphs(args, df, metric, file_name):
	count_vals = get_counts(df)
	if args.family:
		cols = ['bert','baseline','opennmt']
	if args.bert:
		cols = ['layer' + str(i) for i in range(1, 13)]
	df = pd.DataFrame(np.array([count_vals]), columns=cols)
	df = pd.melt(df.reset_index(), id_vars=["index"], value_vars=df.columns, var_name='model', value_name='count') 
	plt.clf()
	sns.set(style="darkgrid")
	if args.aal:
		plt.figure(figsize=(24, 9))
		plt.xticks(rotation=90)
	else:
		plt.figure(figsize=(10, 9))
	g = sns.barplot(x="model", y="count", data=df, ci=68)
	plt.savefig("../visualizations/bayesian_nested_" + str(file_name) + "_count_barplot_" + str(metric) + ".png", bbox_inches='tight')
	return

def plot_bors(bors, file_name):
	plt.clf()
	sns.set(style="darkgrid")
	plt.figure(figsize=(16, 9))
	_ = plt.hist(bors, bins='auto')
	plt.ylabel("count")
	plt.xlabel("BOR")
	plt.savefig("../visualizations/" + str(file_name) + "_hist.png", bbox_inches='tight')

def main():
	parser = argparse.ArgumentParser("calculate nested cv family model")
	parser.add_argument("-family",  "--family", action='store_true', default=False, help="True if use RSA family")
	parser.add_argument("-bert",  "--bert", action='store_true', default=False, help="True if bert")
	parser.add_argument("-aal",  "--aal", action='store_true', default=False, help="True if use RSA aal")
	parser.add_argument("-local",  "--local", action='store_true', default=False, help="True if local")
	parser.add_argument("-save_to_matlab",  "--save_to_matlab", action='store_true', default=False, help="True if save to matlab")
	args = parser.parse_args()

	if args.family and args.bert:
		print("error: choose either family or bert")
		exit(1)

	if not args.family and not args.bert:
		print("error: choose at least family or bert")
		exit(1)

	subjects = [1,2,4,5,7,8,9,10,11]
	print("finding common space...")
	common_space = helper.load_common_space(subjects, local=args.local)
	voxel_coordinates = np.transpose(np.nonzero(common_space))
	print("COMMON SPACE: " + str(common_space.shape))
	print("VOXEL COORDINATES: " + str(voxel_coordinates.shape))

	volmask, num_regions, labels, vals, file_name = helper.get_voxel_labels(args)

	vals_3d = helper.convert_np_to_matlab(vals, volmask)
	labels_vals = vals_3d[np.nonzero(common_space)]

	# get bayesian values
	print("concatenating files...")
	bor, pxp, family = concatenate_files(args)
	print("BOR SHAPE: " + str(bor.shape))
	print("PXP SHAPE: " + str(pxp.shape))
	print("FAMILY SHAPE: " + str(family.shape))

	if args.bert:
		file_name = "bert_only_" + file_name 

	if args.save_to_matlab and args.bert:
		best_layers = np.argmax(pxp, axis=1)
		print("BEST_LAYERS SHAPE: " + str(best_layers.shape))
		vals_3d = helper.convert_np_to_matlab(best_layers, common_space)
		scipy.io.savemat("../nested_bert_layer.mat", dict(metric = vals_3d))

	# get significant values
	bor_3d = helper.convert_np_to_matlab(bor, common_space)
	# plot_bors(bor[~np.isnan(bor)], "all_bor")
	sig_pvals_05 = (np.array(bor) < 0.05).astype(bool)
	# print("bor_3d: " + str(bor_3d.shape))
	# print("sig_pvals_05: " + str(sig_pvals_05.shape))
	# sig05 = bor[sig_pvals_05]
	# sig05vals = sig05[np.nonzero(sig05)]
	# plot_bors(sig05vals[~np.isnan(sig05vals)], "significant_bor05")
	# sig_pvals_05 = (np.array(bor) < 1).astype(bool)
	print("SIG PVALS SHAPE:" + str(sig_pvals_05.shape))

	print("aggregating...")
	# get values per region
	for region in tqdm(range(1, num_regions + 1)):
		indices_bool = (labels_vals == region).astype(bool)
		# print("INDICES SHAPE: " + str(indices_bool.shape))
		# print("NUM IN REGION: " + str(np.sum(indices_bool)))
		sig_indices_bol = np.array(indices_bool) & np.array(sig_pvals_05)
		# print("SIG INDICES SHAPE: " + str(sig_indices_bol.shape))
		# print("SiG NUM IN REGION: " + str(np.sum(sig_indices_bol)))
		indices = np.where(sig_indices_bol == True)[0]
		if args.family:
			region_vals = np.take(family, indices, axis=0) 
			# print("DF REGION: " + str(region_vals.shape))
		if args.bert:
			all_model_region_vals = np.take(pxp, indices, axis=0) 
			region_vals = all_model_region_vals[:, :12]
			# print("DF REGION: " + str(region_vals.shape))
		# if args.bert:
		# 	region_vals = np.take(pxp, indices, axis=0) 
		plot_count_graphs(args, region_vals, "llh", file_name + str(labels[region-1]))

	print("done.")

if __name__ == "__main__":
	main()