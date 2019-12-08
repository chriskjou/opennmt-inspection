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

def all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, num, total_batches, brain_to_model, do_cross_validation, radius=5, kfold_split=5):
	print("getting activations for all sentences...")
	# per_sentence = []
	res_per_spotlight = []
	predictions = []
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
		res, pred = linear_model(embed_matrix, spotlights, do_cross_validation, kfold_split, brain_to_model)
		print("RES for SPOTLIGHT #", index, ": ", res)
		res_per_spotlight.append(res)
		# predictions.append(pred)
		index+=1
		## DECODING ABOVE

	return res_per_spotlight, predictions

def linear_model(embed_matrix, spotlight_activations, do_cross_validation, kfold_split, brain_to_model):
	predicted = []
	if brain_to_model:
		from_regress = np.array(spotlight_activations)
		to_regress = np.array(embed_matrix)
	else:
		from_regress = np.array(embed_matrix)
		to_regress = np.array(spotlight_activations)

	if do_cross_validation:
		kf = KFold(n_splits=kfold_split)
		errors = []
		predicted_trials = []
		for train_index, test_index in kf.split(from_regress):
			X_train, X_test = from_regress[train_index], from_regress[test_index]
			y_train, y_test = to_regress[train_index], to_regress[test_index]
			p, res, rnk, s = lstsq(X_train, y_train)
			residuals = np.sqrt(np.sum((y_test - np.dot(X_test, p))**2))
			# predicted_trials.append(np.dot(X_test, p))
			errors.append(residuals)
		# predicted.append(np.mean(predicted_trials, axis=0))
		return np.mean(errors), predicted
	p, res, rnk, s = lstsq(from_regress, to_regress)
	# predicted.append(np.dot(from_regress, p))
	residuals = np.sqrt(np.sum((to_regress - np.dot(from_regress, p))**2))
	print("RESIDUALS: " + str(residuals))
	print("PREDICTED: " + str(predicted))
	return residuals, predicted

def get_embed_matrix(embedding):
	dict_keys = list(embedding.keys())[3:]
	embed_matrix = np.array([embedding[i][0][1:] for i in dict_keys])
	in_training_bools = np.array([embedding[i][0][0] for i in dict_keys])
	return embed_matrix

# normalize voxels across all sentences per participant
def normalize_voxels(activations):
	avg = np.mean(activations, axis=0)
	std = np.std(activations, axis=0)
	modified_act = (activations - avg)/std
	return modified_act

def main():
	argparser = argparse.ArgumentParser(description="Decoding (linear reg). step for correlating NN and brain")
	argparser.add_argument('--embedding_layer', type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("--subject_mat_file", type=str, help=".mat file ")
	argparser.add_argument("--brain_to_model", type=str, default="False", help="True if regressing brain to model, False if regressing model to brain")
	argparser.add_argument("--cross_validation", type=str, default="False", help="True if add cross validation, False if not")
	argparser.add_argument("--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	argparser.add_argument("--batch_num", type=int, help="batch number of total (for scripting) (out of --total_batches)", required=True)
	argparser.add_argument("--total_batches", type=int, help="total number of batches", required=True)
	argparser.add_argument("--random",  action='store_true', default=False, help="True if initialize random brain activations, False if not")
	argparser.add_argument("--rand_embed",  action='store_true', default=False, help="True if initialize random embeddings, False if not")
	argparser.add_argument("--glove",  action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	argparser.add_argument("--word2vec",  action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("--bert",  action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("--normalize",  action='store_true', default=False, help="True if add normalization across voxels, False if not")
	args = argparser.parse_args()

	if not args.glove and not args.word2vec and not args.bert and not args.rand_embed:
		embed_loc = args.embedding_layer
		file_name = embed_loc.split("/")[-1].split(".")[0]
		embedding = scipy.io.loadmat(embed_loc)
		embed_matrix = get_embed_matrix(embedding)
	else:
		embed_loc = args.embedding_layer
		file_name = embed_loc.split("/")[-1].split(".")[0].split("-")[-1] # aggregation type
		if args.word2vec:
			# embed_matrix = pickle.load( open( "../embeddings/word2vec/" + str(file_name) + ".p", "rb" ) )	
			embed_matrix = pickle.load( open( "/n/scratchlfs/shieber_lab/users/cjou/embeddings/word2vec/" + str(file_name) + ".p", "rb" ) )	
		elif args.glove:
			# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
			embed_matrix = pickle.load( open( "/n/scratchlfs/shieber_lab/users/cjou/embeddings/glove/" + str(file_name) + ".p", "rb" ) )	
		elif args.bert:
			# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
			embed_matrix = pickle.load( open( "/n/scratchlfs/shieber_lab/users/cjou/embeddings/bert/" + str(file_name) + ".p", "rb" ) )
		else: # args.rand_embed
			# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
			embed_matrix = pickle.load( open( "/n/scratchlfs/shieber_lab/users/cjou/embeddings/random/" + str(file_name) + ".p", "rb" ) )	

	# info = sys.argv[2]
	# title = sys.argv[3]
	subj_num = args.subject_number
	num = args.batch_num
	total_batches = args.total_batches

	# file name adjustments
	if args.brain_to_model == "True":
		direction = "brain2model_"
		brain_to_model = True
	else:
		direction = "model2brain_"
		brain_to_model = False

	if args.cross_validation == "True":
		validate = "cv_"
		cross_validation = True
	else:
		validate = "nocv_"
		cross_validation = False

	if args.random:
		rlabel = "random"
	else:
		rlabel = ""

	if args.rand_embed:
		elabel = "rand_embed"
	else:
		elabel = ""

	if args.glove:
		glabel = "glove"
	else:
		glabel = ""

	if args.word2vec:
		w2vlabel = "word2vec"
	else:
		w2vlabel = ""

	if args.bert:
		bertlabel = "bert"
	else:
		bertlabel = ""

	# get modified activations
	activations = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{subj_num}/activations.p", "rb" ) )
	volmask = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{subj_num}/volmask.p", "rb" ) )
	modified_activations = pickle.load( open( f"/n/scratchlfs/shieber_lab/users/fmri/subj{subj_num}/modified_activations.p", "rb" ) )

	if args.normalize:
		modified_activations = normalize_voxels(modified_activations)

	if args.random:
		print("RANDOM ACTIVATIONS")
		modified_activations = np.random.randint(-20, high=20, size=(240, 79, 95, 68))

	all_residuals, predictions = all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, num, total_batches, brain_to_model, cross_validation)
	
	# make file path
	if not os.path.exists('../../projects/residuals/'):
		os.makedirs('../../projects/residuals/')

	if not os.path.exists('../../projects/predictions/'):
		os.makedirs('../../projects/predictions/')

	altered_file_name = "../../projects/residuals/" + str(rlabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj" + str(args.subject_number) + "-" + str(file_name) + "_residuals_part" + str(num) + "of" + str(total_batches) + ".p"
	pickle.dump( all_residuals, open(altered_file_name, "wb" ) )

	# altered_file_name = "../../projects/predictions/" + str(rlabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj" + str(args.subject_number) + "-" + str(file_name) + "_residuals_part" + str(num) + "of" + str(total_batches)
	# pickle.dump( predictions, open(altered_file_name+"-decoding-predictions.p", "wb" ) )
	print("done.")

	### RUN SIGNIFICANT TESTS BELOW

	### RUN SIGNIFICANCE TESTS ABOVE

	return

if __name__ == "__main__":
    main()
