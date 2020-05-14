import scipy.io
from tqdm import tqdm
import pickle
import numpy as np
import sys
import math
from scipy.linalg import lstsq
from sklearn.model_selection import KFold, permutation_test_score
import argparse
import os
import helper
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
import scipy.stats as stats
# import statsmodels.api as sm

def all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, args, kfold_split=5, alpha=1):
	global temp_file_name

	print("getting activations for all sentences...")
	res_per_spotlight = []
	predictions = []
	rankings = []
	llhs = []
	a,b,c = volmask.shape
	nonzero_pts = np.transpose(np.nonzero(volmask))
	true_spotlights = []
	# CHUNK = helper.chunkify(nonzero_pts, args.batch_num, args.total_batches)
	# CHUNK_SIZE = len(CHUNK)

	# iterate over spotlight
	print("for language region ...")
	num_regions = 201
	labels = scipy.io.loadmat("../neurosynth_labels.mat")["initial"] 
	modified_activations = np.array(modified_activations) # (numsize, dim1, dim2, dim3)

	# reset nan as 0
	# for sent in modified_activations:
	# 	sent[np.isnan(sent)] = 0

	nn_matrix = calculate_dist_matrix(embed_matrix) if args.rsa else None 
	for region in tqdm(range(num_regions)):
		print("REGION: " + str(region))
		region_mask = (labels == region)
		spotlights = np.array([act[np.nonzero(region_mask * volmask)] for act in modified_activations])
		spotlights[np.isnan(spotlights)] = 0

		## DECODING BELOW
		if args.rsa: 
			res = rsa(nn_matrix, spotlights)
		else: 
			res, pred, llh, rank = linear_model(embed_matrix, spotlights, args, kfold_split, alpha)
			predictions.append(pred)
			llhs.append(llh)
			rankings.append(rank)

		print("RES for REGION #", region, ": ", res)
		# print("RANK : " + str(rank))
		res_per_spotlight.append(res)
		
		## DECODING ABOVE

	return res_per_spotlight, llhs, rankings #predictions, true_spotlights,  #boolean_masks

def calculate_dist_matrix(matrix_embeddings): 
	n = matrix_embeddings.shape[0]
	mat = np.zeros(shape=(n*(n-1)//2,))
	cosine_sim = lambda x, y: np.dot(x, y) / (np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2))
	it = 0
	for i in range(n): 
		for j in range(i):
			mat[it] = cosine_sim(matrix_embeddings[i], matrix_embeddings[j]) 
			it += 1
	return mat 

def rsa(embed_matrix, spotlights): 
	spotlight_mat = calculate_dist_matrix(spotlights)
	corr, _ = spearmanr(spotlight_mat, embed_matrix)
	return corr

def find_log_pdf(arr, sigmas):
	val = stats.norm.logpdf(arr, 0, sigmas)
	# return np.ma.masked_invalid(val).sum()
	return np.nansum(val)

def vectorize_llh(pred, data, sigmas):
	residuals = np.subtract(data, pred)
	llh = np.sum(np.apply_along_axis(find_log_pdf, 1, residuals, sigmas))
	# print("LLH: " + str(llh))
	return llh

def linear_model(embed_matrix, spotlight_activations, args, kfold_split, alpha):
	global predicted_trials

	predicted = []
	if args.brain_to_model:
		from_regress = np.array(spotlight_activations)
		to_regress = np.array(embed_matrix)
	else:
		from_regress = np.array(embed_matrix)
		to_regress = np.array(spotlight_activations)

	if args.cross_validation:
		kf = KFold(n_splits=kfold_split)
		errors = []
		predicted_trials = np.zeros((to_regress.shape[0], to_regress.shape[1]))
		llhs = []
		rankings = []
		pvalues = []

		if args.add_bias:
			from_regress = helper.add_bias(from_regress)

		if args.permutation:
			np.random.shuffle(from_regress)

		# alphas = np.logspace(-10, 1, 11, endpoint=False)
		# clf = RidgeCV(alphas=alphas).fit(from_regress, to_regress)
		# best_alpha = clf.alpha_
		# print("BEST ALPHA: " + str(best_alpha))
		best_alpha = 0

		# if args.significance:
		# 	clf = Ridge(alpha=best_alpha)
		# 	score, permutation_scores, pvalue = permutation_test_score(clf, from_regress, to_regress, scoring="neg_mean_squared_error", cv=5, n_permutations=100, n_jobs=1)
		# 	pvalues.append(pvalue)

		for train_index, test_index in kf.split(from_regress):
			greatest_possible_rank = len(test_index)

			X_train, X_test = from_regress[train_index], from_regress[test_index]
			y_train, y_test = to_regress[train_index], to_regress[test_index]

			# with ridge regression
			clf = Ridge(alpha=best_alpha)
			clf.fit(X_train, y_train)
			y_hat_test = clf.predict(X_test)
			predicted_trials[test_index] = y_hat_test

			if args.llh:
				y_hat_train = clf.predict(X_train)
				sigma_train = np.sum((y_hat_train - y_train)**2, axis=0)
				llh = vectorize_llh(y_hat_test, y_test, sigma_train)
				llhs.append(llh)

			if args.ranking and args.model_to_brain:
				true_distances = helper.calculate_true_distances(y_hat_test, y_test)
				distance_matrix = helper.compute_distance_matrix(y_hat_test, y_test)
				rank = helper.calculate_rank(true_distances, distance_matrix)
				rank_accuracy = 1 - (rank - 1) * 1.0 / (greatest_possible_rank - 1)
				rankings.append(rank_accuracy)
		errors = np.sqrt(np.sum(np.abs(np.array(predicted_trials) - to_regress)))
		return errors.astype(np.float32), predicted_trials, np.mean(llhs).astype(np.float64), np.mean(rankings).astype(np.float32)
	return

def main():
	global temp_file_name

	argparser = argparse.ArgumentParser(description="Decoding (linear reg). step for correlating NN and brain")
	argparser.add_argument('--embedding_layer', type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("--rsa", action='store_true', default=False, help="True if RSA is used to generate residual values")
	argparser.add_argument("--subject_mat_file", type=str, help=".mat file ")
	argparser.add_argument("--brain_to_model", action='store_true', default=False, help="True if regressing brain to model, False if not")
	argparser.add_argument("--model_to_brain", action='store_true', default=False, help="True if regressing model to brain, False if not")
	argparser.add_argument("--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	argparser.add_argument("--cross_validation", action='store_true', default=True, help="True if add cross validation, False if not")
	argparser.add_argument("--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	# argparser.add_argument("--batch_num", type=int, help="batch number of total (for scripting) (out of --total_batches)", required=True)
	# argparser.add_argument("--total_batches", type=int, help="total number of batches", required=True)
	argparser.add_argument("--random",  action='store_true', default=False, help="True if initialize random brain activations, False if not")
	argparser.add_argument("--rand_embed",  action='store_true', default=False, help="True if initialize random embeddings, False if not")
	argparser.add_argument("--glove",  action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	argparser.add_argument("--word2vec",  action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("--bert",  action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("--normalize",  action='store_true', default=True, help="True if add normalization across voxels, False if not")
	argparser.add_argument("--permutation",  action='store_true', default=False, help="True if permutation, False if not")
	argparser.add_argument("--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	argparser.add_argument("--add_bias",  action='store_true', default=True, help="True if add bias, False if not")
	argparser.add_argument("--llh",  action='store_true', default=True, help="True if calculate likelihood, False if not")
	argparser.add_argument("--ranking",  action='store_true', default=True, help="True if calculate ranking, False if not")
	argparser.add_argument("--mixed_effects",  action='store_true', default=False, help="True if calculate mixed effects, False if not")
	argparser.add_argument("--significance",  action='store_true', default=False, help="True if calculate significance, False if not")
	args = argparser.parse_args()

	if not args.glove and not args.word2vec and not args.bert and not args.rand_embed:
		embed_loc = args.embedding_layer
		file_name = embed_loc.split("/")[-1].split(".")[0]
		embedding = scipy.io.loadmat(embed_loc)
		embed_matrix = helper.get_embed_matrix(embedding)
	else:
		embed_loc = args.embedding_layer
		file_name = embed_loc.split("/")[-1].split(".")[0].split("-")[-1] + "_layer" + str(args.which_layer) # aggregation type + which layer
		embed_matrix = np.array(pickle.load( open( embed_loc , "rb" ) ))

	subj_num = args.subject_number

	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

	# get modified activations
	activations = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{subj_num}/activations.p", "rb" ) )
	volmask = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{subj_num}/volmask.p", "rb" ) )
	modified_activations = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{subj_num}/modified_activations.p", "rb" ) )

	print("PERMUTATION: " + str(args.permutation))
	print("PERMUTATION REGION: " + str(args.permutation_region))

	print("PLABEL: " + str(plabel))
	print("PRLABEL:  " + str(prlabel))

	if args.normalize:
		modified_activations = helper.z_score(modified_activations)
		embed_matrix = helper.z_score(embed_matrix)

	if args.random:
		print("RANDOM ACTIVATIONS")
		modified_activations = np.random.randint(-20, high=20, size=(240, 79, 95, 68))

	# make file path
	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/residuals_od32/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/residuals_od32/')

	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/rsa_neurosynth/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/rsa_neurosynth/')

	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/llh/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/llh/')

	temp_file_name = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj" + str(args.subject_number) + "-" + str(file_name)
	all_residuals, llhs, rankings = all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, args)

	# dump
	if args.rsa:
		file_name = "/n/shieber_lab/Lab/users/cjou/rsa_neurosynth/" + str(temp_file_name) + ".p"
		pickle.dump( all_residuals, open(file_name, "wb" ) )
	
	else:
		if args.llh:
			llh_file_name = "/n/shieber_lab/Lab/users/cjou/llh/" + temp_file_name
			print("LLH SPOTLIGHTS FILE: " + str(llh_file_name))
			pickle.dump( llhs, open(llh_file_name+"-llh.p", "wb" ), protocol=-1 )

		altered_file_name = "/n/shieber_lab/Lab/users/cjou/residuals_od32/" +  temp_file_name
		print("RESIDUALS FILE: " + str(altered_file_name))
		pickle.dump( all_residuals, open(altered_file_name + ".p", "wb" ), protocol=-1 )

		if args.model_to_brain and args.ranking:
			ranking_file_name = "/n/shieber_lab/Lab/users/cjou/final_rankings/" +  temp_file_name
			print("RANKING FILE: " + str(ranking_file_name))
			pickle.dump( rankings, open(ranking_file_name + ".p", "wb" ), protocol=-1 )

	print("done.")

	return

if __name__ == "__main__":
	main()