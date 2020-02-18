import scipy.io
from tqdm import tqdm
import pickle
import numpy as np
import sys
import math
from scipy.linalg import lstsq
from sklearn.model_selection import KFold
from scipy.stats import spearmanr 
import argparse
import os
import helper

def chunkify(lst, num, total):
	if len(lst) % total == 0:
		chunk_size = len(lst) // total
	else:
		chunk_size = len(lst) // total + 1
	start = num * chunk_size
	if num != total - 1:
		end = num * chunk_size + chunk_size
	else:
		end = len(lst)
	return lst[start:end]

def all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, args, radius=5, kfold_split=5):
	print("getting activations for all sentences...")
	# per_sentence = []
	res_per_spotlight = []
	a,b,c = volmask.shape
	nonzero_pts = np.transpose(np.nonzero(volmask))

	# iterate over spotlight
	print("for each spotlight...")

	index=0
	nn_matrix = calculate_dist_matrix(embed_matrix) if args.rsa else None 
	for pt in tqdm(chunkify(nonzero_pts, args.batch_num, args.total_batches)):

		# SPHERE MASK BELOW
		sphere_mask = np.zeros((a,b,c))
		x1,y1,z1 = pt
		for i in range(-radius, radius+1):
			for j in range(-radius, radius+1):
				for k in range(-radius, radius+1):
					xp = x1 + i
					yp = y1 + j
					zp = z1 + k
					pt2 = [xp,yp,zp]
					if 0 <= xp < a and 0 <= yp < b and 0 <= zp < c:
						dist = math.sqrt(i ** 2 + j ** 2 + k ** 2)
						if pt2 in nonzero_pts and dist <= radius:
							sphere_mask[x1+i][y1+j][z1+k] = 1
		# SPHERE MASK ABOVE

		spotlights = []

		# iterate over each sentence
		for sentence_act in modified_activations:
			spot = sentence_act[sphere_mask.astype(bool)]
			remove_nan = np.nan_to_num(spot)
			spotlights.append(remove_nan)

		## DECODING BELOW
		if not args.rsa: 
			res = linear_model(embed_matrix, spotlights, args, kfold_split)
		else: 
			res = rsa(nn_matrix, np.array(spotlights))
		print("RES for SPOTLIGHT #", index, ": ", res)
		res_per_spotlight.append(res)
		index+=1
		## DECODING ABOVE

	return res_per_spotlight
 

def standardize(X): 
	return np.nan_to_num((X - np.mean(X, axis=0)) / np.std(X, axis=0))

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
	# random_noise = np.random.normal(size=embed_matrix.shape)
	# r_corr_e, _ = spearmanr(embed_matrix, random_noise)
	# r_corr_s, _ = spearmanr(spotlight_mat, random_noise)
	# print(f"RANDOM NOISES: {r_corr_e}, {r_corr_s}")
	# print(embed_matrix, spotlight_mat)
	# exit()
	corr, _ = spearmanr(spotlight_mat, embed_matrix)
	return corr

def linear_model(embed_matrix, spotlight_activations, args, kfold_split):
	if brain_to_model:
		from_regress = np.array(spotlight_activations)
		to_regress = embed_matrix
	else:
		from_regress = embed_matrix
		to_regress = np.array(spotlight_activations)

	if do_cross_validation:
		kf = KFold(n_splits=kfold_split)
		errors = []
		for train_index, test_index in kf.split(from_regress):
			X_train, X_test = from_regress[train_index], from_regress[test_index]
			y_train, y_test = to_regress[train_index], to_regress[test_index]
			p, res, rnk, s = lstsq(X_train, y_train)
			residuals = np.sqrt(np.sum((y_test - np.dot(X_test, p))**2))
			errors.append(residuals)
		return np.mean(errors)
	p, res, rnk, s = lstsq(from_regress, to_regress)
	residuals = np.sqrt(np.sum((to_regress - np.dot(from_regress, p))**2))
	return residuals

def get_embed_matrix(embedding):
	dict_keys = list(embedding.keys())[3:]
	embed_matrix = np.array([embedding[i][0][1:] for i in dict_keys])
	in_training_bools = np.array([embedding[i][0][0] for i in dict_keys])
	return embed_matrix

def main():
	argparser = argparse.ArgumentParser(description="Decoding (linear reg). step for correlating NN and brain")
	argparser.add_argument("--rsa", action='store_true', default=False, help="True if RSA is used to generate residual values")
	argparser.add_argument('--embedding_layer', type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("--subject_mat_file", type=str, help=".mat file ")
	argparser.add_argument("--brain_to_model", action='store_true', default=False, help="True if regressing brain to model, False if not")
	argparser.add_argument("--model_to_brain", action='store_true', default=False, help="True if regressing model to brain, False if not")
	argparser.add_argument("--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	argparser.add_argument("--cross_validation", action='store_true', default=False, help="True if add cross validation, False if not")
	argparser.add_argument("--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	argparser.add_argument("--batch_num", type=int, help="batch number of total (for scripting) (out of --total_batches)", required=True)
	argparser.add_argument("--total_batches", type=int, help="total number of batches", required=True)
	argparser.add_argument("--random",  action='store_true', default=False, help="True if initialize random brain activations, False if not")
	argparser.add_argument("--rand_embed",  action='store_true', default=False, help="True if initialize random embeddings, False if not")
	argparser.add_argument("--glove",  action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	argparser.add_argument("--word2vec",  action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("--bert",  action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("--normalize",  action='store_true', default=False, help="True if add normalization across voxels, False if not")
	argparser.add_argument("--permutation",  action='store_true', default=False, help="True if permutation, False if not")
	argparser.add_argument("--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	argparser.add_argument("--memmap",  action='store_true', default=False, help="True if memmep, False if not")
	args = argparser.parse_args()

	embed_loc = args.embedding_layer
	file_name = embed_loc.split("/")[-1].split(".")[0]
	embedding = scipy.io.loadmat(embed_loc)
	embed_matrix = get_embed_matrix(embedding)
	# info = sys.argv[2]
	# title = sys.argv[3]
	subj_num = args.subject_number
	num = args.batch_num
	total_batches = args.total_batches

	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

	# get modified activations
	activations = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{subj_num}/" + str(plabel) + str(prlabel) + "activations.p", "rb" ) )
	volmask = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{subj_num}/volmask.p", "rb" ) )
	modified_activations = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{subj_num}/" + str(plabel) + str(prlabel) + "modified_activations.p", "rb" ) )


	all_residuals = all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, args)
	
	# make file path
	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/rsa/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/rsa/')

	file_name = "/n/shieber_lab/Lab/users/cjou/rsa/" + str(direction) + str(validate) + str(file_name) + "_residuals_part" + str(num) + "of" + str(total_batches) + ".p"
	pickle.dump( all_residuals, open(file_name, "wb" ) )
	print("done.")

	### RUN SIGNIFICANT TESTS BELOW

	### RUN SIGNIFICANCE TESTS ABOVE

	return

if __name__ == "__main__":
    main()
