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
import scipy.stats as stats

def get_modified_activations(activations, volmask):
	i,j,k = volmask.shape
	nonzero_pts = np.transpose(np.nonzero(volmask))
	modified_activations = []
	for sentence_activation in tqdm(activations):
		one_sentence_act = np.zeros((i,j,k))
		for pt in range(len(nonzero_pts)):
			x,y,z = nonzero_pts[pt]
			one_sentence_act[int(x)][int(y)][int(z)] = sentence_activation[pt]
		modified_activations.append(one_sentence_act)
	return modified_activations

def run_per_voxel(df, labels, conditional_labels):
	y_predicted_all = np.zeros((df.shape[0],))
	for train_index, test_index in kf.split(df):
		greatest_possible_rank = len(test_index)

		X_train, X_test = from_regress[train_index], from_regress[test_index]
		y_train, y_test = to_regress[train_index], to_regress[test_index]

		training_data, testing_data = train_test_split(df, test_size=0.2)

		md = smf.mixedlm('embedding ~ 1 + ' + str(labels) + ' + (1 + ' + str(conditional_labels) + ' )', training_data, groups=training_data["subject_number"])
		mdf = md.fit()
		print(mdf.summary())

		y_hat_test = mdf.predict(testing_data)
		y_predicted_all[test_index] = y_hat_test

	y_true = df['activations']
	rmse = np.sqrt(np.sum(np.abs(y_hat_test - y_true)))
	return rmse.astype(np.float32)

def mixed_effects_analysis(args, embed_matrix):
	# load common brain space
	subjects = [1,2,4,5,7,8,9,10,11]
	common_space = helper.load_common_space(subjects, local=args.local)
	voxel_coordinates = np.transpose(np.nonzero(common_space))
	num_voxels = len(voxel_coordinates)
	print("NUM VOXELS IN SHARED COMMON BRAIN SPACE: " + str(num_voxels))

	# initialize variables
	all_activations = []
	subj_number = []
	voxel_index = []

	# prepare model embeddings 
	dim_labels = ['dim'+str(i) for i in range(embed_matrix.shape[1])]
	embed_matrix_pd = pd.DataFrame(embed_matrix, columns=dim_labels)
	embed_matrix_pd_repeat = pd.concat([embed_matrix_pd]*len(subjects), ignore_index=True)
	print("LENGTH OF EMBEDDINGS: " + str(len(embed_matrix_pd_repeat)))

	# get labels
	labels = ""
	conditional_labels = ""
	for i in range(embed_matrix.shape[1]):
		labels += 'dim' + str(i) + ' '
		conditional_labels += 'dim' + str(i) + ' | subject_number '

	# get data
	for subj in subjects:
		activation = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/activations.p", "rb" ) )
		activation_vals = activation[np.nonzero(common_space)]
		modified_activations = get_modified_activations(activation_vals, common_space)
		all_activations.append(modified_activations)
		voxel_index.append(range(num_voxels))
		subj_number.extend([subj] * num_voxels)
	
	# create dataframe
	data = pd.DataFrame({
		'subject_number': subj_number,
		'voxel_index': voxel_index,
		'activations': all_activations
		})

	data_slice = data.iloc[data["voxel_index"] == 0]
	print("DATA SLICE LENGTH: " + str(len(data_slice)))

	# per voxel
	rmses_per_voxel = []
	for v in range(num_voxels):
		data_slice = data.iloc[data["voxel_index"] == v]
		concat_pd = pd.concat([data_slice, embed_matrix_pd_repeat], axis=1)
		rmse = run_per_voxel(concat_pd, labels, conditional_labels)
		rmses_per_voxel.append(rmse)
		
	return rmses_per_voxel

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

	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

	# get modified activations
	activations = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/activations.p", "rb" ) )
	volmask = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/volmask.p", "rb" ) )
	modified_activations = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/modified_activations.p", "rb" ) )

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

	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/predictions_od32/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/predictions_od32/')

	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/')

	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/rsa/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/rsa/')

	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/llh/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/llh/')

	temp_file_name = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj" + str(args.subject_number) + "-" + str(file_name) + "_no_spotlight"
	
	# get residuals and predictions
	# all_residuals, predictions, true_spotlights, llhs = all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, args)
	
	if args.mixed_effects:
		val = mixed_effects_analysis(args, embed_matrix)
	else:
		all_residuals, llhs, rankings = all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, args)

	# dump
	if args.rsa:
		file_name = "/n/shieber_lab/Lab/users/cjou/rsa/" + str(temp_file_name) + ".p"
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
