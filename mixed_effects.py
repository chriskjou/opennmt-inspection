import scipy.io
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import sys
import math
from sklearn.model_selection import KFold
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

def get_activations(modified_activations):
	return modified_activations[np.nonzero(modified_activations)]

def run_per_voxel(df, from_regress, labels):
	y_predicted_all = np.zeros((df.shape[0],))
	kf = KFold(n_splits=5, shuffle=True)
	for train_index, test_index in kf.split(df):

		training_X = from_regress.loc[train_index,]
		training_y = df.loc[train_index,]['activations']
		training_y_groups = df.loc[train_index,]['subject_number']
		testing_X = from_regress.loc[test_index,]
		testing_y = df.loc[test_index,]['activations']
		testing_y_groups = df.loc[test_index,]['subject_number']

		md = sm.MixedLM(endog=training_y, exog=training_X, groups=training_y_groups)
		# func = 'activations ~ ' + str(labels) + '1'
		# print(func)
		# print(training_data.columns.values.tolist())
		# md = smf.mixedlm(func, training_data, groups=training_data["subject_number"])
		mdf = md.fit()
		# print(mdf.summary())

		# print(testing_y.shape)
		y_hat_test = mdf.predict(testing_X)
		y_predicted_all[test_index] = y_hat_test
		# print(y_hat_test.shape)
		# print(np.sqrt(np.sum(np.abs(y_hat_test - testing_y))))
		# print(asdf)

	y_true = df['activations']
	rmse = np.sqrt(np.sum(np.abs(y_hat_test - y_true)))
	return rmse.astype(np.float32)

def mixed_effects_analysis(args, embed_matrix):
	# load common brain space
	subjects = [1,2,4,5,7,8,9,10,11]
	num_sentences = 240
	common_space = helper.load_common_space(subjects, local=args.local)
	print("COMMON SPACE SHAPE: " + str(common_space.shape))
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
	print("EMBEDDINGS SHAPE: " + str(embed_matrix_pd.shape))
	embed_matrix_pd_repeat = pd.concat([embed_matrix_pd]*len(subjects), ignore_index=True)
	embed_matrix_pd_repeat.insert(0, 'bias', 1)
	print("REPEAT EMBEDDINGS SHAPE: " + str(embed_matrix_pd_repeat.shape))

	# get labels
	labels = ""
	conditional_labels = ""
	for i in range(embed_matrix.shape[1]):
		labels += 'dim' + str(i) + ' + '
		conditional_labels += 'dim' + str(i) + ' | subject_number + '

	# get data
	for subj in tqdm(subjects):
		if args.local:
			modified_activations = pickle.load( open( f"../examplesGLM/subj{subj}/modified_activations.p", "rb" ) )
		else:
			modified_activations = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{subj}/modified_activations.p", "rb" ) )
		
		norm_modified_activations = helper.z_score(np.array(modified_activations))
		activation_vals = np.array([modified_elem[np.nonzero(common_space)] for modified_elem in norm_modified_activations])
		# print("ACTIVATIONS SHAPE: " + str(activation_vals.shape))
		flatten_activations = get_activations(activation_vals)
		# print("FLATTEN ACTIVATIONS SHAPE: " + str(flatten_activations.shape))
		all_activations.extend(flatten_activations)
		voxel_index.extend(list(range(num_voxels)) * num_sentences)
		subj_number.extend([subj] * num_voxels * num_sentences)
		del modified_activations
		del norm_modified_activations
		del activation_vals
		del flatten_activations

	print("ACTIVATIONS LENGTH: " + str(len(all_activations)))
	print("SUBJECT NUMBER LENGTH: " + str(len(subj_number)))
	print("VOXEL INDEX: " + str(len(voxel_index)))
	
	# create dataframe
	data = pd.DataFrame({
		'subject_number': subj_number,
		'voxel_index': voxel_index,
		'activations': all_activations
		})

	data_slice = data.loc[data["voxel_index"] == 0]
	print("DATA SLICE SHAPE: " + str(data_slice.shape))

	# per voxel
	rmses_per_voxel = []
	CHUNK = helper.chunkify(list(range(num_voxels)), args.batch_num, args.total_batches)
	for v in tqdm(CHUNK):
		data_slice = data.loc[data["voxel_index"] == v].reset_index()
		# concat_pd = pd.concat([data_slice, embed_matrix_pd_repeat], axis=1)
		rmse = run_per_voxel(data_slice, embed_matrix_pd_repeat, labels)
		rmses_per_voxel.append(rmse)
		
	return rmses_per_voxel

def main():
	global temp_file_name

	argparser = argparse.ArgumentParser(description="Decoding (linear reg). step for correlating NN and brain")
	argparser.add_argument('--embedding_layer', type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("--rsa", action='store_true', default=False, help="True if RSA is used to generate residual values")
	argparser.add_argument("--subject_mat_file", type=str, help=".mat file ")
	argparser.add_argument("--brain_to_model", action='store_true', default=False, help="True if regressing brain to model, False if not")
	argparser.add_argument("--model_to_brain", action='store_true', default=True, help="True if regressing model to brain, False if not")
	argparser.add_argument("--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	argparser.add_argument("--cross_validation", action='store_true', default=True, help="True if add cross validation, False if not")
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
	argparser.add_argument("--mixed_effects",  action='store_true', default=True, help="True if calculate mixed effects, False if not")
	argparser.add_argument("--local",  action='store_true', default=False, help="True if local, False if not")
	argparser.add_argument("--batch_num", type=int, help="batch number of total (for scripting) (out of --total_batches)", required=True)
	argparser.add_argument("--total_batches", type=int, help="total number of batches", default=100)
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

	print("PERMUTATION: " + str(args.permutation))
	print("PERMUTATION REGION: " + str(args.permutation_region))

	print("PLABEL: " + str(plabel))
	print("PRLABEL:  " + str(prlabel))

	# normalize
	embed_matrix = helper.z_score(embed_matrix)

	# make file path
	if args.local: 
		if not os.path.exists('../mixed_effects/'):
			os.makedirs('../mixed_effects/')
	else:
		if not os.path.exists('/n/shieber_lab/Lab/users/cjou/mixed_effects/'):
			os.makedirs('/n/shieber_lab/Lab/users/cjou/mixed_effects/')

	temp_file_name = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-" + str(file_name) + "_mixed_effects"
	
	# get residuals and predictions
	# all_residuals, predictions, true_spotlights, llhs = all_activations_for_all_sentences(modified_activations, volmask, embed_matrix, args)
	
	rmses = mixed_effects_analysis(args, embed_matrix)

	# dump
	# if args.llh:
	# 	llh_file_name = "/n/shieber_lab/Lab/users/cjou/llh/" + temp_file_name
	# 	print("LLH SPOTLIGHTS FILE: " + str(llh_file_name))
	# 	pickle.dump( llhs, open(llh_file_name+"-llh.p", "wb" ), protocol=-1 )

	altered_file_name = "../mixed_effects/" +  temp_file_name
	print("RESIDUALS FILE: " + str(altered_file_name))
	pickle.dump( rmses, open(altered_file_name + ".p", "wb" ), protocol=-1 )

	# if args.model_to_brain and args.ranking:
	# 	ranking_file_name = "/n/shieber_lab/Lab/users/cjou/final_rankings/" +  temp_file_name
	# 	print("RANKING FILE: " + str(ranking_file_name))
	# 	pickle.dump( rankings, open(ranking_file_name + ".p", "wb" ), protocol=-1 )

	print("done.")

	return

if __name__ == "__main__":
	main()
