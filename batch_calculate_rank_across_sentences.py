import numpy as np
import argparse
from tqdm import tqdm
import pickle
import scipy.io
import os
import math
import time
import gc
import helper
from scipy.spatial import distance

def calculate_true_distances(a, b):
	return np.linalg.norm(a - b, axis=1)
	# return np.sqrt(np.sum((a-b)**2))

def compute_distance_matrix(a, b):
	return distance.cdist(a, b, 'euclidean')

def calculate_rank(true_distance, distance_matrix):
	num_sentences, dim = distance_matrix.shape

	ranks = []
	for sent_index in range(num_sentences):
		distances = distance_matrix[sent_index]
		# print("ALL DISTANCES: " + str(distances))
		true_sent_distance = true_distance[sent_index]
		# print("TRUE DISTANCE: " + str(true_sent_distance))
		rank = np.sum(distances < true_sent_distance)
		ranks.append(rank)

	return np.mean(ranks)

def get_file_name(args, specific_file, i, true_activations=False):
	file_name = specific_file + "_residuals_part" + str(i) + "of" + str(args.total_batches) 
	if not true_activations:
		if args.local:
			file_path = "../predictions_od32/"
		else:
			file_path = "/n/shieber_lab/Lab/users/cjou/predictions_od32/"
		# print("FILE NAME: " + file_path+ file_name + "-decoding-predictions.p")
		return file_path + file_name + "-decoding-predictions.p"
	if args.local:
		file_path = "../true_spotlights_od32/"
	else:
		file_path = "/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/"
	# print("FILE NAME: " + file_path + file_name + "-true-spotlights.p")
	return file_path + file_name + "-true-spotlights.p"

def calculate_average_rank(args, file_name):

	final_rankings = []

	for i in tqdm(range(args.total_batches)):
		# print("batch num: " + str(i))
		spotlight_activations_file_name = get_file_name(args, file_name, i, true_activations=True)
		spotlight_predictions_file_name = get_file_name(args, file_name, i)
		spotlight_activations = pickle.load(open(spotlight_activations_file_name, "rb"))
		spotlight_predictions = pickle.load(open(spotlight_predictions_file_name, "rb"))

		num_voxels = len(spotlight_activations)
		num_voxels_check = len(spotlight_predictions)

		# print("number of voxels: " + str(num_voxels))
		# print("number of voxels check:" + str(num_voxels_check))

		if num_voxels != num_voxels_check:
			print("unequal number of voxels")
			exit()

		for j in range(num_voxels):
			if not np.array_equal(np.array(spotlight_predictions[j]).shape, np.array(spotlight_activations[j]).shape):
				print("not same size")
				print(np.array(spotlight_predictions[j]).shape)
				print(np.array(spotlight_activations[j]).shape)
				exit()
				# final_rankings.append(0)
			# print("shape: " + str(np.array(spotlight_predictions[j]).shape))
			# print(str(np.array(spotlight_predictions[j])))
			# print(str(np.array(spotlight_activations[j])))
			true_distances = calculate_true_distances(np.array(spotlight_predictions[j]), np.array(spotlight_activations[j]))
			# print("true distnaces: " + str(true_distances))
			# print("NUM of true distances: " + str(len(true_distances)))
			distance_matrix = compute_distance_matrix(np.array(spotlight_predictions[j]), np.array(spotlight_activations[j]))
			# print("distance_matrix: " + str(distance_matrix))
			# print("SHAPE matrix: " + str(distance_matrix.shape))
			# print("possible distances: " + str(len(distances)))
			rank = calculate_rank(true_distances, distance_matrix)
			# print("rank: " + str(rank))
			final_rankings.append(rank)

		del spotlight_activations
		del spotlight_predictions

	to_save_file = "/n/shieber_lab/Lab/users/cjou/final_rankings/" + file_name + ".p"
	pickle.dump(final_rankings, open(to_save_file, "wb"))
	return 

def main():
	argparser = argparse.ArgumentParser(description="calculate rankings for model-to-brain")
	# argparser.add_argument("-embedding_layer", "--embedding_layer", type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("-total_batches", "--total_batches", type=int, help="total number of batches residual_name is spread across", required=True)
	argparser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	argparser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	argparser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	argparser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	argparser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation", action='store_true', default=False)
	argparser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model", action='store_true', default=False)
	argparser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain", action='store_true', default=False)
	argparser.add_argument("-glove", "--glove", action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	argparser.add_argument("-word2vec", "--word2vec", action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("-bert", "--bert", action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("-rand_embed", "--rand_embed", action='store_true', default=False, help="True if initialize random embeddings, False if not")
	argparser.add_argument("-random",  "--random", action='store_true', default=False, help="True if add cross validation, False if not")
	argparser.add_argument("-permutation",  "--permutation", action='store_true', default=False, help="True if permutation, False if not")
	argparser.add_argument("-permutation_region", "--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	argparser.add_argument("-normalize", "--normalize",  action='store_true', default=False, help="True if add normalization across voxels, False if not")
	argparser.add_argument("-gpu", "--gpu",  action='store_true', default=False, help="True if gpu False if not")
	argparser.add_argument("-local", "--local",  action='store_true', default=False, help="True if local False if not")
	args = argparser.parse_args()

	# check conditions // can remove when making pipeline
	if args.brain_to_model and args.model_to_brain:
		print("select only one flag for brain_to_model or model_to_brain")
		exit()
	if not args.brain_to_model and not args.model_to_brain:
		print("select at least flag for brain_to_model or model_to_brain")
		exit()

	if args.brain_to_model:
		print("not valid for ranking experiment")
		exit()

	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

	if not args.local:
		if not os.path.exists('/n/shieber_lab/Lab/users/cjou/final_rankings/'):
			os.makedirs('/n/shieber_lab/Lab/users/cjou/final_rankings/')

	if not args.bert and not args.glove and not args.word2vec:
		specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"	
		file_name = specific_file.format(
			args.subject_number, 
			args.language, 
			args.num_layers, 
			args.model_type, 
			args.which_layer, 
			args.agg_type
		)
	else:
		file_name = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-{}_layer{}".format(args.subject_number, args.agg_type, args.which_layer)

	### BRAIN ACTIVATIONS BELOW ###
	# volmask = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/volmask.p", "rb" ) )
	# modified_activations = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/" + str(plabel) + str(prlabel) + "modified_activations.p", "rb" ) )

	# if args.normalize:
	# 	modified_activations = normalize_voxels(modified_activations)

	# if args.random:
	# 	print("RANDOM ACTIVATIONS")
	# 	modified_activations = np.random.randint(-20, high=20, size=(240, 79, 95, 68))
	### BRAIN ACTIVATIONS ABOVE ###

	print("calculating average rank...")
	start = time.time()
	calculate_average_rank(args, file_name)
	end = time.time()
	print("time: " + str(end-start)) 
	print("done.")

if __name__ == "__main__":
	main()