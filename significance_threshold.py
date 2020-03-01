from tqdm import tqdm
import scipy.io
import pickle
import numpy as np
import sys
from sklearn.model_selection import KFold
import argparse
import os
from scipy.stats import pearsonr
from scipy import stats
from scipy.linalg import lstsq
import random
import math
import statsmodels.stats.multitest as smm
import matplotlib.pyplot as plt
import helper
plt.switch_backend('agg')

def get_embed_matrix(embedding, num_sentences=240):
	embed_matrix = np.array([embedding["sentence" + str(i+1)][0][1:] for i in range(num_sentences)])
	in_training_bools = np.array([embedding["sentence" + str(i+1)][0][0] for i in range(num_sentences)])
	return embed_matrix

def z_score(matrix):
	mean = np.mean(matrix, axis=0)
	std = np.std(matrix, axis=0)
	z_score = (matrix - mean) / std
	return z_score

def generate_indices(index, session, num_scanner_runs=3):
	indices = [i * num_scanner_runs + s for i in index for s in session]
	return np.array(indices)

def check_for_nan_infinity(y, index):
	indx = np.argwhere(~np.isnan(y[:,index]) & np.isfinite(y[:,index]))
	return np.reshape(indx, (len(indx), ))

def calculate_pearson_correlation(args, activations, embeddings, kfold_split=5, num_scanner_runs=3, split_scanner_runs=False):
	num_sentences = embeddings.shape[0]
	num_voxels = activations.shape[1]
	print("num sentences: " + str(num_sentences))
	print("num voxels: " + str(num_voxels))

	test_session = [random.choice(range(num_scanner_runs))]
	train_session = list(set(range(num_scanner_runs)) - set(test_session))

	kf = KFold(n_splits=kfold_split)
	correlations = {}
	pvals = {}

	for train_index, test_index in tqdm(kf.split(range(num_sentences))):
		# create data
		training_indices = generate_indices(train_index, train_session)
		testing_indices = generate_indices(test_index, test_session)
		y_train = activations[training_indices]
		y_test = activations[testing_indices]

		X_train = np.array(list(embeddings[train_index]) * len(train_session))
		X_test = np.array(list(embeddings[test_index]) * len(test_session))

		# GLM per voxel
		for voxel_index in range(num_voxels):
			train_valid_index = check_for_nan_infinity(y_train, voxel_index)
			test_valid_index = check_for_nan_infinity(y_test, voxel_index)
			if len(train_valid_index) == 0:
				correlations.setdefault(voxel_index, []).append(0)
				pvals.setdefault(voxel_index, []).append(0)
			else:
				corr, pval = fit_to_GLM(args, X_train[train_valid_index], X_test[test_valid_index], y_train[:,voxel_index][train_valid_index], y_test[:,voxel_index][test_valid_index])
				correlations.setdefault(voxel_index, []).append(np.arctanh(corr))
				pvals.setdefault(voxel_index, []).append(pval)

	return correlations, pvals

def fit_to_GLM(args, X_train, X_test, y_train, y_test):
	p, res, rnk, s = lstsq(X_train, y_train)
	predicted = np.dot(X_test, p)
	corr, pval = pearsonr(y_test, predicted)
	return corr, pval

def plot_pvals(pvals):
	N = len(pvals)
	q = 0.05
	p_values = np.sort(pvals)
	i = np.arange(1, N+1)
	plt.clf()
	plt.xlabel('$i$')
	plt.ylabel('p value')
	plt.plot(i, p_values, 'b.', label='$p(i)$')
	plt.plot(i, q * i / N, 'r', label='$q i / N$')
	plt.xlabel('$i$')
	plt.ylabel('$p$')
	plt.legend()
	plt.savefig("FDR_correction")

def fdr_correction(correlations, pvals):
	N = len(pvals)
	q = 0.05
	i = np.arange(1, N+1)
	below = pvals < (q * i / N)
	max_below = np.max(np.where(below)[0])
	valid_correlations = np.array(correlations)[below.astype(bool)]
	print(valid_correlations)
	indices = np.where(below == False)
	return valid_correlations, indices[0]

	# print('p_i:' + str(pvals[max_below]))
	# print('i:' + str(max_below + 1))
	# print(below)
	# print(correlations)

def get_pval_from_ttest(correlations):
	pvals = []
	for voxel_index in sorted(correlations):
		t, prob = stats.ttest_1samp(correlations[voxel_index], 0.0)
		pvals.append(prob)
	return pvals, len(correlations)

def average_correlations(correlations):
	avg_correlations = []
	for index in sorted(correlations):
		avg_corr = np.mean(correlations[index])
		avg_correlations.append(avg_corr)
	return avg_correlations

def get_spotlights(volmask):
	volmask_shape = volmask.shape
	nonzero_pts = np.transpose(np.nonzero(volmask))
	tuple_matching = [tuple(a) for a in nonzero_pts]  
	space_to_index_dict = dict(zip(tuple_matching, range(len(nonzero_pts))))
	index_to_space_dict = dict(zip(range(len(nonzero_pts)), tuple_matching))
	return space_to_index_dict, index_to_space_dict, volmask_shape

def get_correlations_in_spotlight(correlations, space_to_index_dict, index_to_space_dict, volmask_shape, radius=1):
	num_voxels = len(spotlight_dict)

	correlations_in_spotlight = []
	a,b,c = volmask_shape
	for voxel in tqdm(range(num_voxels)):
		curr_voxel_in_space = index_to_space_dict[voxel]
		x1, y1, z1 = curr_voxel_in_space

		for i in range(-radius, radius+1):
			for j in range(-radius, radius+1):
				for k in range(-radius, radius+1):
					xp = x1 + i
					yp = y1 + j
					zp = z1 + k
					pt2 = (xp,yp,zp)
					if 0 <= xp and 0 <= yp and 0 <= zp and xp < a and yp < b and zp < c:
						dist = math.sqrt(i ** 2 + j ** 2 + k ** 2)
						if pt2 in space_to_index_dict and dist <= radius:
							correlations_in_spotlight.setdefault(voxel, []).append(correlations[pt2])
	return correlations_in_spotlight

def evaluate_performance(args, correlations, pvals_per_voxel, space_to_index_dict, index_to_space_dict, volmask_shape):
	### get pvals
	if args.searchlight:
		correlations_in_spotlight = get_correlations_in_spotlight(correlations, space_to_index_dict, index_to_space_dict, volmask_shape)
		pvals, num_voxels = get_pval_from_ttest(correlations_in_spotlight)
	else:
		pvals, num_voxels = get_pval_from_ttest(correlations)
	# plot_pvals(pvals)

	### get average correlations
	if args.searchlight:
		correlations_in_spotlight = get_correlations_in_spotlight(correlations, space_to_index_dict, index_to_space_dict, volmask_shape)
		avg_correlations = average_correlations(correlations_in_spotlight)
	else:
		avg_correlations = average_correlations(correlations)

	if args.fdr:
		valid_correlations, indices = fdr_correction(avg_correlations, pvals)
		return valid_correlations, indices, num_voxels

	return avg_correlations, range(len(avg_correlations)), num_voxels


def get_2d_coordinates(correlations, indices, num_voxels):
	arr = np.zeros((num_voxels,))
	np.put(arr, indices, correlations)
	return arr

def fix_coords_to_absolute_value(coords):
	norm_coords = [c if c==0 else c+1 for c in coords]
	return norm_coords

def get_volmask(subj_num):
	volmask = pickle.load(open(f"/n/shieber_lab/Lab/users/cjou/fmri/subj" + str(subj_num) + "/volmask.p", "rb"))
	return volmask

def load_common_space(subject_numbers):
	subject_volmasks = get_volmask(subject_numbers[0])
	for subj_num in subject_numbers[1:]:
		volmask = get_volmask(subj_num)
		subject_volmasks = subject_volmasks & volmask
	return subject_volmasks

def main():
	argparser = argparse.ArgumentParser(description="FDR significance thresholding for single subject")
	argparser.add_argument("-embedding_layer", "--embedding_layer", type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("-subject_number", "--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	argparser.add_argument("-random", "--random",  action='store_true', default=False, help="True if initialize random brain activations, False if not")
	argparser.add_argument("-rand_embed", "--rand_embed",  action='store_true', default=False, help="True if initialize random embeddings, False if not")
	argparser.add_argument("-glove", "--glove",  action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	argparser.add_argument("-word2vec", "--word2vec",  action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("-bert", "--bert",  action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("-normalize", "--normalize",  action='store_true', default=False, help="True if add normalization across voxels, False if not")
	argparser.add_argument("-permutation", "--permutation",  action='store_true', default=False, help="True if permutation, False if not")
	argparser.add_argument("-permutation_region", "--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	argparser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	argparser.add_argument("-single_subject", "--single_subject", help="if single subject analysis", action='store_true', default=False)
	argparser.add_argument("-group_level", "--group_level", help="if group level analysis", action='store_true', default=False)
	argparser.add_argument("-searchlight", "--searchlight", help="if searchlight", action='store_true', default=False)
	argparser.add_argument("-fdr", "--fdr", help="if apply FDR", action='store_true', default=False)
	argparser.add_argument("-subjects", "--subjects", help="subject numbers", type=str, default="")
	args = argparser.parse_args()

	### check conditions
	if not args.single_subject and not args.group_level:
		print("select analysis type: single subject or group level")
		exit()

	if args.fdr and args.single_subject and not args.searchlight:
		print("not valid application of FDR to single subject with searchlight")
		exit()

	if args.group_level and args.subject == "":
		print("must specify subject numbers in group level analysis")
		exit()

	### get embeddings
	if not args.glove and not args.word2vec and not args.bert and not args.rand_embed:
		embed_loc = args.embedding_layer
		# embed_loc = "/Users/christinejou/Documents/research/embeddings/parallel/spanish/2layer-brnn/avg/parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg.mat"
		file_name = embed_loc.split("/")[-1].split(".")[0]
		embedding = scipy.io.loadmat(embed_loc)
		embed_matrix = get_embed_matrix(embedding)
	else:
		embed_loc = args.embedding_layer
		file_name = embed_loc.split("/")[-1].split(".")[0].split("-")[-1] + "_layer" + str(args.which_layer) # aggregation type + which layer
		embed_matrix = pickle.load( open( embed_loc , "rb" ) )
		if args.word2vec:
			file_name += "word2vec"
		elif args.glove:
			file_name += "glove"
		elif args.bert:
			file_name += "bert"
		else:
			file_name += "random"

	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/mat/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/mat/')

	if args.single_subject:
		save_location = "/n/shieber_lab/Lab/users/cjou/fdr/" + str(file_name) + "_subj" + str(args.subject_number)
		volmask = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj" + str(args.subject_number) + "/volmask.p", "rb" ) )
		space_to_index_dict, index_to_space_dict, volmask_shape = get_spotlights(volmask)

		# 1. z-score
		print("z-scoring activations and embeddings...")
		individual_activations = pickle.load(open("../../examplesGLM/subj" + str(args.subject_number) + "/individual_activations.p", "rb"))
		z_activations = z_score(individual_activations)
		z_embeddings = z_score(embed_matrix)

		# 2. calculate correlation 
		print("calculating correlations...")
		correlations, pvals = calculate_pearson_correlation(args, z_activations, z_embeddings)
		
		# 3. evaluate significance
		print("evaluating significance...")
		valid_correlations, indices, num_voxels = evaluate_performance(args, correlations, pvals, space_to_index_dict, index_to_space_dict, volmask_shape)
		corrected_coordinates = get_2d_coordinates(valid_correlations, indices, num_voxels)
		norm_coords = fix_coords_to_absolute_value(corrected_coordinates)
		_ = helper.transform_coordinates(norm_coords, volmask, save_location, "fdr")
		print("done.")

	if args.group_level:
		save_location = "/n/shieber_lab/Lab/users/cjou/fdr/" + str(file_name) + "_group_analysis"
		subject_numbers = [int(subj_num) for subj_num in args.subjects.split(",")]   
		volmask = load_common_space(subject_numbers)
		space_to_index_dict, index_to_space_dict, volmask_shape = get_spotlights(volmask)

		# 1. z-score
		print("z-scoring activations and embeddings...")
		z_embeddings = z_score(embed_matrix)
		z_activations = []
		for subj_num in subject_numbers:
			individual_activations = pickle.load(open("../../examplesGLM/subj" + str(subj_num) + "/individual_activations.p", "rb"))
			z_activations.append(z_score(individual_activations))

		# 2. calculate correlation 
		print("calculating correlations...")
		correlations, pvals = calculate_pearson_correlation(args, z_activations, z_embeddings)
		
		# 3. evaluate significance
		print("evaluating significance...")
		valid_correlations, indices, num_voxels = evaluate_performance(args, correlations, pvals, space_to_index_dict, index_to_space_dict, volmask_shape)
		corrected_coordinates = get_2d_coordinates(valid_correlations, indices, num_voxels)
		norm_coords = fix_coords_to_absolute_value(corrected_coordinates)
		_ = helper.transform_coordinates(norm_coords, volmask, save_location, "fdr")
		print("done.")
	return

if __name__ == "__main__":
	main()