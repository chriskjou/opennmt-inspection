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

	for train_index, test_index in kf.split(range(num_sentences)):
		# create data
		training_indices = generate_indices(train_index, train_session)
		testing_indices = generate_indices(test_index, test_session)
		y_train = activations[training_indices]
		y_test = activations[testing_indices]

		X_train = np.array(list(embeddings[train_index]) * len(train_session))
		X_test = np.array(list(embeddings[test_index]) * len(test_session))

		# GLM per voxel
		for voxel_index in tqdm(range(num_voxels)):
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
	plt.savefig("FDR correction")

def get_bounds(correlations, pvals):
	N = len(pvals)
	q = 0.05
	below = pvals < (q * i / N)
	max_below = np.max(np.where(below)[0])
	print('p_i:' + str(pvals[max_below]))
	print('i:' + str(max_below + 1))
	valid_correlations = correlations[below]
	return valid_correlations

def get_pval_from_ttest(pvals_per_voxel):
	pvals = []
	for voxel_index in sorted(pvals_per_voxel):
		t, prob = stats.ttest_1samp(pvals_per_voxel[voxel_index], 0.0)
		pvals.append(voxel_index)
	return pvals

def evaluate_performance(correlations, pvals_per_voxel):
	pvals = get_pval_from_ttest(pvals_per_voxel)
	min_index = plot_pvals(pvals)
	valid_correlations = get_bounds(correlations, pval_dict)
	return valid_correlations

def main():
	argparser = argparse.ArgumentParser(description="FDR significance thresholding")
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
	args = argparser.parse_args()

	### get embeddings
	if not args.glove and not args.word2vec and not args.bert and not args.rand_embed:
		embed_loc = args.embedding_layer
		# embed_loc = "/Users/christinejou/Documents/research/embeddings/parallel/spanish/2layer-brnn/avg/parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg.mat"
		file_name = embed_loc.split("/")[-1].split(".")[0]
		embedding = scipy.io.loadmat(embed_loc)
		embed_matrix = get_embed_matrix(embedding)
	else:
		embed_loc = args.embedding_layer
		file_name = embed_loc.split("/")[-1].split(".")[0].split("-")[-1] # aggregation type
		if args.word2vec:
			# embed_matrix = pickle.load( open( "../embeddings/word2vec/" + str(file_name) + ".p", "rb" ) )	
			embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/word2vec/" + str(file_name) + ".p", "rb" ) )	
		elif args.glove:
			# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
			embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/glove/" + str(file_name) + ".p", "rb" ) )	
		elif args.bert:
			# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
			embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/bert/" + str(file_name) + ".p", "rb" ) )
		else: # args.rand_embed
			# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
			embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/rand_embed/rand_embed.p", "rb" ) )	
	
	# 1. z-score
	print("z-scoring activations and embeddings...")
	individual_activations = pickle.load(open("../examplesGLM/subj" + str(args.subject_number) + "/individual_activations.p", "rb"))
	z_activations = z_score(individual_activations)
	z_embeddings = z_score(embed_matrix)

	# 2. calculate correlation 
	print("calculating correlations...")
	correlations, pvals = calculate_pearson_correlation(args, z_activations, z_embeddings)
	
	# 3. evaluate significance
	save_location = "../../"
	print("evaluating significance...")
	valid_correlations = evaluate_performance(correlations, pvals)
	pickle.dump(open(save_location+"valid_correlations.p", "wb"))
	print("done.")

	return

if __name__ == "__main__":
	main()