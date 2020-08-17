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

def get_values(args, subj_num):
	all_corrs = []
	for layer in tqdm(range(1, args.num_layers+1)):
		file_name = "bert{}{}-subj{}-{}_layer{}".format(
						"model2brain_",
						"cv_",
						subj_num,
						"avg",
						layer
					)
		if args.local:
			folder = "../mat_rsa_cos_nonnorm/"
			folder = "../mat_rsa/"
		else:
			folder = "/n/home09/cjou/projects/mat/"
		content = scipy.io.loadmat(str(folder) + str(file_name) + "-3dtransform-rsa.mat")["metric"]
		all_corrs.append(content)
	return np.array(all_corrs)

def calculate_slope(args, all_corrs):
	dims = all_corrs[0].shape
	slopes = np.zeros((dims[0], dims[1], dims[2]))
	num_layers = 12
	
	for i in range(dims[0]):
		for j in range(dims[1]):
			for k in range(dims[2]):
				vals_across_layers = []
				for layer in range(num_layers):
					vals_across_layers.append(all_corrs[layer][i][j][k])
				if args.slope:
					if np.sum(vals_across_layers) == 0:
						slopes[i][j][k] = 0
					else:
						slope, intercept, r_value, p_value, std_err = linregress(list(zip(vals_across_layers, list(range(1, num_layers+1)))))
						slopes[i][j][k] = slope
				if args.argmax:
					max_index = np.argmax(vals_across_layers)
					if np.sum(vals_across_layers) == 0:
						slopes[i][j][k] = np.nan
					else:
						slopes[i][j][k] = max_index + 1
	return slopes

def calculate_slope_per_null(args, all_corrs):
	num_layers = all_corrs.shape[0]
	num_regions = all_corrs.shape[1]
	slopes = np.zeros(num_regions)
	
	for i in range(num_regions):
		vals_across_layers = []
		for layer in range(num_layers):
			vals_across_layers.append(all_corrs[layer][i])
		if args.slope:
			vals_across_layers = np.array(vals_across_layers)
			vals_across_layers = vals_across_layers[~np.isnan(vals_across_layers)]
			if np.sum(vals_across_layers) == 0:
				slopes[i] = 0
			else:
				slope, intercept, r_value, p_value, std_err = linregress(list(zip(vals_across_layers, list(range(1, num_layers+1)))))
				slopes[i] = slope
		if args.argmax:
			max_index = np.argmax(vals_across_layers)
			if np.sum(vals_across_layers) == 0:
				slopes[i] = np.nan
			else:
				slopes[i] = max_index + 1
	return slopes

def get_values_per_null(args, subj_num):
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
		true_std = np.array(pickle.load(open("../" + str(file_name)+ temp_file_name + "_std.p", "rb")))

		standardized = (vals - true_mean) / true_std

		region_vals.append(standardized)
	return np.array(region_vals)

def calculate_ttest(args, slopes):
	dims = slopes[0].shape
	vals_across_subjs = np.zeros((dims[0], dims[1], dims[2]))
	pvals = np.zeros((dims[0], dims[1], dims[2]))
	subjects = [1,2,4,5,7,8,9,10,11]

	for i in range(dims[0]):
		for j in range(dims[1]):
			for k in range(dims[2]):
				vals_across_subjs = []
				for subj in range(len(subjects)):
					vals_across_subjs.append(slopes[subj][i][j][k])
				if args.slope:
					null_hypo = np.zeros((len(vals_across_subjs),))
				else:
					null_hypo = np.full((len(vals_across_subjs),),  6.5)
				tstat, pval = ttest_ind(null_hypo, vals_across_subjs)
				pvals[i][j][k] = pval
	return pvals

def calculate_ttest_per_null(args, slopes):
	num_subjs = len(slopes)
	num_regions = slopes[0].shape[0]
	vals_across_subjs = np.zeros((num_subjs, num_regions))
	pvals = np.zeros(num_regions)
	subjects = [1,2,4,5,7,8,9,10,11]

	for i in range(num_regions):
		vals_across_subjs = []
		for subj in range(len(subjects)):
			vals_across_subjs.append(slopes[subj][i])
		if args.slope:
			null_hypo = np.zeros((len(vals_across_subjs),))
		else:
			null_hypo = np.full((len(vals_across_subjs),),  6.5)
		tstat, pval = ttest_ind(null_hypo, vals_across_subjs)
		pvals[i] = pval
	return pvals

def calculate_anova(args, all_corrs):
	dims = all_corrs[0][0].shape
	pvals = np.zeros((dims[0], dims[1], dims[2]))
	num_layers = 12
	num_subjs = 9
	print("LEN: " + str(len(all_corrs)))
	print("DIMS: " + str(all_corrs[0][0].shape))

	for i in tqdm(range(dims[0])):
		for j in range(dims[1]):
			for k in range(dims[2]):

				vals_across_subjs_and_layers = []
				for subj in range(num_subjs):
					for layer in range(num_layers):
						val = all_corrs[subj][layer][i][j][k]
						vals_across_subjs_and_layers.append(all_corrs[subj][layer][i][j][k])
				
				# make dataframe
				df = pd.DataFrame({
					'voxel': np.ones(len(vals_across_subjs_and_layers)),
					'corr': vals_across_subjs_and_layers,
					'subject': np.repeat(list(range(1, num_subjs+1)), num_layers),
					'layer': np.tile(list(range(1, num_layers+1)), num_subjs) 
				})

				aovrm2way = AnovaRM(df, 'voxel', 'corr', within=['subject', 'layer'])
				mod = aovrm2way.fit()
				pval = mod.summary().tables[0]["Pr > F"]["subject:layer"]
				pvals[i][j][k] = pval
	return pvals

def plot_graphs(values, xlabel, file_name):
	plt.clf()
	sns.set(style="darkgrid")
	plt.figure(figsize=(16, 9))
	_ = plt.hist(values, bins='auto')
	plt.ylabel("count")
	plt.xlabel(xlabel)
	plt.savefig(str(file_name) + ".png", bbox_inches='tight')
	return

def plot_histograms(args, metric, file_name, slope_avgs, pvals):
	# before
	plot_vals = slope_avgs[~np.isnan(slope_avgs)]
	plot_pvals = pvals[~np.isnan(pvals)]

	if args.slope:
		plot_vals = plot_vals[np.nonzero(plot_vals)]

	plot_graphs(plot_vals, metric, "../visualizations/" + str(file_name) + "_before")
	plot_graphs(plot_pvals, "pval", ".../visualizations/" + str(file_name) + "_pval_before")

	# after
	thres1 = slope_avgs * (pvals < 0.1).astype(bool)
	thres05 = slope_avgs * (pvals < 0.05).astype(bool)
	thres01 = slope_avgs * (pvals < 0.01).astype(bool)

	thres1 = thres1[~np.isnan(thres1)]
	thres01 = thres01[~np.isnan(thres01)]
	thres05 = thres05[~np.isnan(thres05)]

	if args.slope:
		thres1 = thres1[np.nonzero(thres1)]
		thres01 = thres01[np.nonzero(thres01)]
		thres05 = thres05[np.nonzero(thres05)]

	if not args.null:
		plot_graphs(thres1, metric, "../visualizations/" + str(file_name) + "after1")
		plot_graphs(thres01, metric, "../visualizations/" + str(file_name) + "after01")
		plot_graphs(thres05, metric, "../visualizations/" + str(file_name) + "after05")
		
	return metric, slope_avgs * (pvals < 0.05).astype(bool)

def plot_across_subjects(args, metric, file_name, thres05):
	print("getting values...")
	vollangloc = scipy.io.loadmat("../subj1_vollangloc.mat")["vollangloc"]
	if args.aal:
		aal_labels = pickle.load( open( "../examplesGLM/subj1/atlas_labels.p", "rb" ) )
		labels = [str(elem[0][0]) for elem in aal_labels]
		print(labels)
	else:
		labels = ['LMidPostTemp', 'LPostTemp', 'LMidAntTemp', 'LIFG', 'LAntTemp', 'LIFGorb', 'LAngG', 'LMFG']

	all_region_vals = []
	all_region_labels = []
	all_layer_labels = []

	for label in range(1, len(labels)+1):
		if args.null:
			vals = thres05[:,[label-1]]
			vals = np.reshape(vals, (len(vals),))
		else:
			roi = thres05 * (vollangloc == label).astype(bool)
			vals = roi[~np.isnan(roi)]
			vals = vals[np.nonzero(vals)]
		all_region_vals.extend(vals)
		all_region_labels.extend([labels[label-1]] * len(vals))

	print("ALL VALS: " + str(len(all_region_vals)))
	print("ALL REGIONS: " + str(len(all_region_labels)))

	# create dataframe
	df = pd.DataFrame({
		metric: all_region_vals,
		'ROI': all_region_labels
		})

	print("DATAFRAME")
	print(df.head())

	print("plotting...")
	plt.clf()
	sns.set(style="darkgrid")
	if args.aal:
		plt.figure(figsize=(24, 9))
		g = sns.barplot(x="ROI", y=metric, data=df, ci=68)
		plt.xticks(rotation=90)
	else:
		plt.figure(figsize=(10, 9))
		g = sns.barplot(x="ROI", y=metric, data=df, ci=68)
	plt.savefig("../visualizations/" + str(file_name) + "_barplot_" + str(metric) + ".png", bbox_inches='tight')
	return

def main():
	parser = argparse.ArgumentParser("calculate slope/argmax maps")
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers", type=int, default=12)
	parser.add_argument("-slope", "--slope", action='store_true', default=False, help="slope map")
	parser.add_argument("-argmax", "--argmax", action='store_true', default=False, help="argmax")
	parser.add_argument("-anova",  "--anova", action='store_true', default=False, help="True if anova")
	# parser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	parser.add_argument("-local",  "--local", action='store_true', default=False, help="True if running locally")
	parser.add_argument("-null",  "--null", action='store_true', default=False, help="True if use RSA region null distribution")
	parser.add_argument("-aal",  "--aal", action='store_true', default=False, help="True if use RSA aal")
	args = parser.parse_args()

	subjects = [1,2,4,5,7,8,9,10,11]
	if not args.slope and not args.argmax:
		print("error: please select at least slope or argmax")
		exit()

	if args.slope and args.argmax:
		print("error: please select only one slop argmax")
		exit()

	all_slopes = []
	all_corrs = []
	slope_avgs = []
	print("getting slopes...")
	for subj_num in tqdm(subjects):
		if args.null:
			corrs = get_values_per_null(args, subj_num)
			slopes = calculate_slope_per_null(args, corrs)
		else:
			corrs = get_values(args, subj_num)
			slopes = calculate_slope(args, corrs)
		all_corrs.append(corrs)
		all_slopes.append(slopes)

	print("for thresholding...")
	slope_avgs = np.mean(all_slopes, axis=0)
	# print(slope_avgs[0])
	# print("BEFORE: " + str(np.count_nonzero(~np.isnan(slope_avgs))))
	# slope_avgs[np.isnan(slope_avgs)] = 0
	# print("AFTER: " + str(np.count_nonzero(~np.isnan(slope_avgs))))

	print("running t-tests...")

	# 1 sample t-test
	if args.anova:
		pvals = calculate_anova(args, all_corrs)
	else:
		if args.null:
			pvals = calculate_ttest_per_null(args, all_slopes)
		else:
			pvals = calculate_ttest(args, all_slopes)

	if args.slope:
		file_name = "cos_rsa_slope"
		metric = "slope"
	else:
		if args.anova:
			file_name = "cos_rsa_argmax_anova"
		else:
			file_name = "cos_rsa_argmax"
		metric = "argmax"

	if args.null:
		file_name += "_null"

	if args.aal:
		file_name += "_aal"

	if args.null:
		plot_across_subjects(args, metric, file_name, np.array(all_slopes))
	else:
		pickle.dump(slope_avgs, open("../cos_visualize_" + file_name + ".p", "wb"))
		pickle.dump(pvals, open("../cos_threshold_" + file_name + ".p", "wb"))

		metric, thres05 = plot_histograms(args, metric, file_name, slope_avgs, pvals)
		plot_across_subjects(args, metric, file_name, thres05)
		
		significant = (pvals < 0.1).astype(bool)
		scipy.io.savemat("../" + str(file_name) + "_across_subjects_1.mat", dict(metric = slope_avgs * significant))
		significant = (pvals < 0.05).astype(bool)
		scipy.io.savemat("../" + str(file_name) + "_across_subjects_05.mat", dict(metric = slope_avgs * significant))

	print("done.")

if __name__ == "__main__":
    main()