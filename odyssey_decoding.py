import scipy.io
from tqdm import tqdm
import pickle
import numpy as np
import sys
import math
from scipy.linalg import lstsq
from sklearn.model_selection import KFold
import argparse
import os

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

def all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, num, total_batches, brain_to_model, radius=5, do_cross_validation=False, kfold_split=5):
	print("getting activations for all sentences...")
	# per_sentence = []
	res_per_spotlight = []
	a,b,c = volmask.shape
	nonzero_pts = np.transpose(np.nonzero(volmask))

	# iterate over spotlight
	print("for each spotlight...")

	index=0
	for pt in tqdm(chunkify(nonzero_pts, num, total_batches)):

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
					if 0 <= xp and 0 <= yp and 0 <= zp and xp < a and yp < b and zp < c:
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
		res = linear_model(embed_matrix, spotlights, do_cross_validation, kfold_split, brain_to_model)
		print("RES for SPOTLIGHT #", index, ": ", res)
		res_per_spotlight.append(res)
		index+=1
		## DECODING ABOVE

	return res_per_spotlight

def linear_model(embed_matrix, spotlight_activations, do_cross_validation, kfold_split, brain_to_model):
	if brain_to_model:
		from_regress = spotlight_activations
		to_regress = embed_matrix
	else:
		from_regress = embed_matrix
		to_regress = spotlight_activations

	if do_cross_validation:
		kf = KFold(n_splits=kfold_split)
		errors = []
		for train_index, test_index in kf.split(X):
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
	argparser = argparse.ArgumentParser(description="Decoding (linear reg). step from NN to brain")
	argparser.add_argument('--embedding_layer', type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("--subject_mat_file", type=str, help=".mat file ")
	argparser.add_argument("--brain_to_model", type=str, default="False", help="True if regressing brain to model, False if regressing model to brain")
	argparser.add_argument("--cross_validation", type=str, default="False", help="True if add cross validation, False if not")
	argparser.add_argument("--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	argparser.add_argument("--batch_num", type=int, help="batch number of total (for scripting) (out of --total_batches)", required=True)
	argparser.add_argument("--total_batches", type=int, help="total number of batches", required=True)
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
	brain_to_model = args.brain_to_model
	cross_validation = args.cross_validation

	# file name adjustments
	if brain_to_model == "True":
		direction = "brain2model_"
	else:
		direction = "model2brain_"

	if cross_validation == "False":
		validate = "cv_"
	else:
		validate = "nocv_"

	# get modified activationscd
	activations = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{subj_num}/activations.p", "rb" ) )
	volmask = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{subj_num}/volmask.p", "rb" ) )
	modified_activations = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{subj_num}/modified_activations.p", "rb" ) )

	all_residuals = all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, num, total_batches, brain_to_model, cross_validation)
	
	# make file path
	if not os.path.exists('../../projects/residuals/'):
		os.makedirs('../../projects/residuals/')

	file_name = "../../projects/residuals/" + str(direction) + str(validate) + str(file_name) + "_residuals_part" + str(num) + "of" + str(total_batches) + ".p"
	pickle.dump( all_residuals, open(file_name, "wb" ) )
	print("done.")

	### RUN SIGNIFICANT TESTS BELOW

	### RUN SIGNIFICANCE TESTS ABOVE

	return

if __name__ == "__main__":
    main()
