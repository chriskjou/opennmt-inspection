import pickle
import argparse
import pandas as pd 
from tqdm import tqdm
import numpy as np 
import helper
import scipy.io
from scipy.stats import linregress, ttest_1samp, ttest_ind
from statsmodels.stats.anova import AnovaRM 

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
		if not args.local:
			folder = "mat/"
		else:
			folder = "mat_rsa_cos_nonnorm/"
		content = scipy.io.loadmat("../" + str(folder) + str(file_name) + "-3dtransform-rsa.mat")["metric"]
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

def main():
	parser = argparse.ArgumentParser("calculate slope maps")
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers", type=int, default=12)
	parser.add_argument("-slope", "--slope", action='store_true', default=False, help="slope map")
	parser.add_argument("-argmax", "--argmax", action='store_true', default=False, help="argmax")
	parser.add_argument("-anova",  "--anova", action='store_true', default=False, help="True if anova")
	# parser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	parser.add_argument("-local",  "--local", action='store_true', default=True, help="True if running locally")
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
		pvals = calculate_ttest(args, all_slopes)

	if args.slope:
		file_name = "cos_rsa_slope"
	else:
		if args.anova:
			file_name = "cos_rsa_argmax_anova"
		else:
			file_name = "cos_rsa_argmax"

	pickle.dump(slope_avgs, open("../cos_visualize_" + file_name + ".p", "wb"))
	pickle.dump(pvals, open("../cos_threshold_" + file_name + ".p", "wb"))
	

	significant = (pvals < 0.1).astype(bool)
	scipy.io.savemat("../" + str(file_name) + "_across_subjects_1.mat", dict(metric = slope_avgs * significant))
	significant = (pvals < 0.05).astype(bool)
	scipy.io.savemat("../" + str(file_name) + "_across_subjects_05.mat", dict(metric = slope_avgs * significant))

	print("done.")

if __name__ == "__main__":
    main()