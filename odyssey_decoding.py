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

def pad_along_axis(array, target_length, axis=0):
	pad_size = target_length - array.shape[axis]
	axis_nb = len(array.shape)
	if pad_size < 0:
		return array
	npad = [(0, 0) for x in range(axis_nb)]
	npad[axis] = (0, pad_size)
	b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)
	return b

def get_voxel_number(args, CHUNK_SIZE, i):
	return args.batch_num * CHUNK_SIZE + i

def get_dimensions(data):
	return int(data[0])+1, int(data[1]), int(data[2])

def all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, args, radius=5, kfold_split=5):
	global temp_file_name
	global memmap_file_name

	ACTIVATION_SHAPE = (240, 515)

	print("getting activations for all sentences...")
	res_per_spotlight = []
	predictions = []
	a,b,c = volmask.shape
	nonzero_pts = np.transpose(np.nonzero(volmask))
	true_spotlights = []
	CHUNK = chunkify(nonzero_pts, args.batch_num, args.total_batches)
	CHUNK_SIZE = len(CHUNK)

	# iterate over spotlight
	print("for each spotlight...")

	if args.memmap:
		predictions_memmap_read = np.memmap("/n/shieber_lab/Lab/users/cjou/predictions_memmap/" + memmap_file_name + ".dat", mode='r', dtype='float32')
		dim1, dim2, dim3 = get_dimensions(predictions_memmap_read)
		del predictions_memmap_read
		predictions_memmap = np.memmap("/n/shieber_lab/Lab/users/cjou/predictions_memmap/" + memmap_file_name + ".dat", dtype='float32', mode='r+', shape=(dim1, dim2, dim3))

		if args.model_to_brain:
			true_spotlights_memmap_read = np.memmap("/n/shieber_lab/Lab/users/cjou/true_spotlights_memmap/" + memmap_file_name + ".dat", mode='r', dtype='float32')
			dim1, dim2, dim3 = get_dimensions(true_spotlights_memmap_read)
			del true_spotlights_memmap_read
			true_spotlights_memmap = np.memmap("/n/shieber_lab/Lab/users/cjou/true_spotlights_memmap/" + memmap_file_name + ".dat", dtype='float32', mode='r+', shape=(dim1, dim2, dim3))

		if args.brain_to_model:
			target_size = embed_matrix.shape
		else:
			target_size = ACTIVATION_SHAPE

	# 	# create
	# 	print("SHAPE: ")
	# 	print(str(CHUNK_SIZE) + " " + str(embed_matrix.shape[0]) + " " + str(embed_matrix.shape[1]))
	# 	predictions_memmap = np.memmap("/n/shieber_lab/Lab/users/cjou/predictions_memmap/" + temp_file_name + ".dat", dtype='float32', mode='w+', shape=(CHUNK_SIZE+1, target_size[0], target_size[1]))
	# 	# add size to predictions and true_spotlights 
	# 	pred_size = np.zeros((1, target_size[0], target_size[1]))
	# 	pred_size[0][0][0] = CHUNK_SIZE
	# 	pred_size[0][0][1] = target_size[0]
	# 	pred_size[0][0][2] = target_size[1]
	# 	predictions_memmap[0] = pred_size
	# 	# del predictions_memmap

	# 	if args.model_to_brain:
	# 		true_spotlights_memmap = np.memmap("/n/shieber_lab/Lab/users/cjou/true_spotlights_memmap/" + temp_file_name + ".dat", dtype='float32', mode='w+', shape=(CHUNK_SIZE+1, target_size[0], target_size[1]))
	# 		true_size = np.zeros((1, target_size[0], target_size[1]))
	# 		true_size[0][0][0] = CHUNK_SIZE
	# 		true_size[0][0][1] = target_size[0]
	# 		true_size[0][0][2] = target_size[1]
	# 		true_spotlights_memmap[0] = true_size
	# del true_spotlights_memmap

	index=0
	nn_matrix = calculate_dist_matrix(embed_matrix) if args.rsa else None 
	for pt in tqdm(CHUNK):

		# SPHERE MASK BELOW
		sphere_mask = np.zeros((a,b,c))
		x1,y1,z1 = pt
		# points_glm.append(pt)
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
		spotlight_mask = []

		# iterate over each sentence
		for sentence_act in modified_activations:
			spot = sentence_act[sphere_mask.astype(bool)]
			remove_nan = np.nan_to_num(spot).astype(np.float32)
			spotlights.append(remove_nan)
			# spotlight_mask.append(sphere_mask.astype(bool))

		print(np.array(spotlights).shape)

		true_spotlights.append(spotlights)
		# boolean_masks.append(spotlight_mask)

		## DECODING BELOW
		if args.rsa: 
			res = rsa(nn_matrix, np.array(spotlights))
		else: 
			res, pred = linear_model(embed_matrix, spotlights, args, kfold_split)
			predictions.append(pred)

		print("RES for SPOTLIGHT #", index, ": ", res)
		res_per_spotlight.append(res)

		index+=1
		# pad remaining
		if args.memmap:
			pred_pad = pad_along_axis(np.array(pred), target_size[1], axis=1)
			index_of_voxel = get_voxel_number(args, CHUNK_SIZE, index)
			print("INDEX OF VOXEL:" + str(index_of_voxel))
			# add to memmap files
			# predictions_memmap = np.load("/n/shieber_lab/Lab/users/cjou/predictions_od32/" + temp_file_name + ".dat", mmap_mode='w')
			predictions_memmap[index_of_voxel] = pred_pad

			# res_pad = pad_along_axis(res, target_size, axis=1)
			if args.model_to_brain:
				true_spotlight_pad = pad_along_axis(np.array(spotlights), target_size[1], axis=1)
				print(true_spotlight_pad.shape)
				# true_spotlights_memmap = np.load("/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/" + temp_file_name + ".dat", mmap_mode='w')
				true_spotlights_memmap[index_of_voxel] = true_spotlight_pad
			print(pred_pad.shape)
		
		## DECODING ABOVE
	if args.memmap:
		del predictions_memmap
		if args.model_to_brain:
			del true_spotlights_memmap

	return res_per_spotlight, predictions, true_spotlights #boolean_masks

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
		predicted_trials = []
		for train_index, test_index in kf.split(from_regress):
			X_train, X_test = from_regress[train_index], from_regress[test_index]
			y_train, y_test = to_regress[train_index], to_regress[test_index]

			if not args.add_bias:
				p, res, rnk, s = lstsq(X_train, y_train)
			else:
				new_col = np.ones((X_train.shape[0], 1))
				with_bias = np.hstack((X_train, new_col))
				p_with_bias, res, rnk, s = lstsq(with_bias, y_train)
				p = p_with_bias[:-1]
			residuals = np.sqrt(np.sum((y_test - np.dot(X_test, p))**2)).astype(np.float32)
			predicted_trials.append(np.dot(from_regress, p))
			errors.append(residuals)
		predicted = np.mean(predicted_trials, axis=0).astype(np.float32)
		# print(rnk.shape)
		return np.mean(errors).astype(np.float32), predicted
	# print("FROM REGRESS: " + str(from_regress.shape))
	# print("TO REGRESS: " + str(to_regress.shape))
	if not args.add_bias:
		p, res, rnk, s = lstsq(from_regress, to_regress)
	else:
		new_col = np.ones((from_regress.shape[0], 1))
		with_bias = np.hstack((to_regress, new_col))
		p_with_bias, res, rnk, s = lstsq(with_bias, to_regress)
		p = p_with_bias[:-1]
	# print("P: " + str(p.shape))
	# print("RES: " + str(res.shape))
	# print("RNK: " + str(np.array(rnk).shape))
	# print("S: " + str(s.shape))
	# print("COMPUTED: " + str(np.dot(from_regress, p).shape))
	# print("EQUAL: " + str(np.array_equal(p, np.dot(from_regress, p))))
	predicted = np.dot(from_regress, p).astype(np.float32)
	residuals = np.sqrt(np.sum((to_regress - np.dot(from_regress, p))**2)).astype(np.float32)
	# print("RESIDUALS: " + str(residuals))
	# print("PREDICTED: " + str(predicted))
	return residuals, predicted

def get_embed_matrix(embedding, num_sentences=240):
	embed_matrix = np.array([embedding["sentence" + str(i+1)][0][1:] for i in range(num_sentences)])
	in_training_bools = np.array([embedding["sentence" + str(i+1)][0][0] for i in range(num_sentences)])
	return embed_matrix

# normalize voxels across all sentences per participant
def normalize_voxels(activations):
	avg = np.mean(activations, axis=0)
	std = np.std(activations, axis=0)
	modified_act = (activations - avg)/std
	return modified_act

def main():
	global temp_file_name
	global memmap_file_name

	argparser = argparse.ArgumentParser(description="Decoding (linear reg). step for correlating NN and brain")
	argparser.add_argument('--embedding_layer', type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("--rsa", action='store_true', default=False, help="True if RSA is used to generate residual values")
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
	argparser.add_argument("--add_bias",  action='store_true', default=False, help="True if add bias, False if not")
	args = argparser.parse_args()

	if not args.glove and not args.word2vec and not args.bert and not args.rand_embed:
		embed_loc = args.embedding_layer
		file_name = embed_loc.split("/")[-1].split(".")[0]
		embedding = scipy.io.loadmat(embed_loc)
		embed_matrix = get_embed_matrix(embedding)
	else:
		embed_loc = args.embedding_layer
		file_name = embed_loc.split("/")[-1].split(".")[0].split("-")[-1] + "_layer" + str(args.which_layer) # aggregation type + which layer
		embed_matrix = pickle.load( open( embed_loc , "rb" ) )
		# if args.word2vec:
			# embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/word2vec/" + str(file_name) + ".p", "rb" ) )	
		# elif args.glove:
			# embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/glove/" + str(file_name) + ".p", "rb" ) )	
		# elif args.bert:
			# embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/bert/" + str(file_name) + ".p", "rb" ) )
		# else: # args.rand_embed
			# embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/rand_embed/rand_embed.p", "rb" ) )	

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

	if args.normalize:
		modified_activations = normalize_voxels(modified_activations)

	if args.random:
		print("RANDOM ACTIVATIONS")
		modified_activations = np.random.randint(-20, high=20, size=(240, 79, 95, 68))

	# make file path
	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/residuals_od32/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/residuals_od32/')

	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/predictions_od32/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/predictions_od32/')

	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/')

	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/rsa/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/rsa/')

	# create memmap
	# temp_file_name = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj" + str(args.subject_number) + "-" + str(file_name)
	
	# if args.batch_num == 0:
	# 	create_memmap_files(args, "/n/shieber_lab/Lab/users/cjou/predictions_od32/", temp_file_name, activations.shape[1], embed_matrix.shape[0], embed_matrix.shape[1])
	# 	create_memmap_files(args, "/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/", temp_file_name, activations.shape[1], embed_matrix.shape[0], embed_matrix.shape[1])
	
	temp_file_name = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj" + str(args.subject_number) + "-" + str(file_name) + "_residuals_part" + str(args.batch_num) + "of" + str(args.total_batches)
	memmap_file_name = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj" + str(args.subject_number) + "-" + str(file_name)

	# get residuals and predictions
	all_residuals, predictions, true_spotlights = all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, args)
	
	# dump
	if not args.memmap and not args.rsa:
		altered_file_name = "/n/shieber_lab/Lab/users/cjou/residuals_od32/" +  temp_file_name
		print("RESIDUALS FILE: " + str(altered_file_name))
		pickle.dump( all_residuals, open(altered_file_name + ".p", "wb" ), protocol=-1 )

		pred_file_name = "/n/shieber_lab/Lab/users/cjou/predictions_od32/" + temp_file_name
		print("PREDICTIONS FILE: " + str(pred_file_name))
		pickle.dump( predictions, open(pred_file_name+"-decoding-predictions.p", "wb" ), protocol=-1 )

		spot_file_name = "/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/" + temp_file_name
		print("TRUE SPOTLIGHTS FILE: " + str(spot_file_name))
		pickle.dump( true_spotlights, open(spot_file_name+"-true-spotlights.p", "wb" ), protocol=-1 )

	if args.rsa:
		file_name = "/n/shieber_lab/Lab/users/cjou/rsa/" + str(temp_file_name)
		pickle.dump( all_residuals, open(file_name, "wb" ) )
	print("done.")

	return

if __name__ == "__main__":
	main()
