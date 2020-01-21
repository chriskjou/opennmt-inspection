import numpy as np
import argparse
from tqdm import tqdm
import pickle
import scipy.io
import os
import math
import time
import multiprocessing as mp
import gc
import helper

spotlight_file_path = "/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/"

def get_embed_matrix(embedding):
	dict_keys = list(embedding.keys())[3:]
	embed_matrix = np.array([embedding[i][0][1:] for i in dict_keys])
	in_training_bools = np.array([embedding[i][0][0] for i in dict_keys])
	return embed_matrix

def calculate_euclidean_distance(a, b):
	return np.sqrt(np.sum((a-b)**2))

def get_data(filename):
	global VOXEL_NUMBER
	fp = np.memmap(filename, dtype='float32', mode='r')
	VOXEL_NUMBER, num_sentences, act = int(fp[0]), int(fp[1]), int(fp[2])
	padding = num_sentences * act
	fp = fp[padding:].reshape((VOXEL_NUMBER, num_sentences, act))
	return fp

def trim_zeros(padded_array):
	return np.apply_along_axis(np.trim_zeros, 1, padded_array)

def get_data(filename):
	global VOXEL_NUMBER
	fp = np.memmap(filename, dtype='float32', mode='r')
	VOXEL_NUMBER, num_sentences, act = int(fp[0]), int(fp[1]), int(fp[2])
	padding = num_sentences * act
	fp = fp[padding:].reshape((VOXEL_NUMBER, num_sentences, act))
	return fp

def get_file_name(args, file_path, specific_file, i, true_activations=False):
	file_name = specific_file + "_residuals_part" + str(i) + "of" + str(args.total_batches) + ".dat"
	return file_path + file_name

def get_true_activations(args, file_path, file_name, pred_index):
	entire_file_name = file_name + "_residuals_part" + str(args.batch_num) + "of" + str(args.total_batches) + ".dat"
	# entire_file_name = file_name + "_residuals_part" + str(args.batch_num) + "of" + str(args.total_batches) + "-true-spotlights.p"
	file_contents = get_data(file_path + entire_file_name)
	return file_contents[pred_index]

def compare_rankings_to_brain(i, true_activations, radius=5):
	global g_args
	global g_pred_contents
	global g_file_name
	global g_predicted_distance

	predictions = g_pred_contents[i]
	rank = 0
	spotlight_activations = g_pred_contents

	true_predicted_distance = calculate_euclidean_distance(np.array(predictions), np.array(true_activations))

	start = time.time()
	for sentence_act in spotlight_activations:
		trimmed_pred = trim_zeros(np.array(true_activations))
		if np.array_equal(trimmed_pred.shape, np.array(sentence_act).shape):
			dist = calculate_euclidean_distance(trimmed_pred, np.array(sentence_act))
			if dist <= true_predicted_distance:
				rank+=1
	end = time.time()
	print("time to get data for batch " + str(i) + ": " + str(end-start))
	
	del spotlight_activations

	return rank

def compare_rankings_to_embeddings(prediction, embeddings):
	return

def calculate_average_rank(args, file_name, embeddings):
	global g_args
	global g_file_name
	global g_pred_contents
	global g_true_activations

	### PREDICTIONS BELOW ###
	file_path = "/n/shieber_lab/Lab/users/cjou/predictions_memmap/"
	### PREDICTIONS ABOVE ###

	### GET PREDICTIONS BELOW ###
	start = time.time()
	file = get_file_name(g_args, file_path, g_file_name, args.batch_num)
	g_pred_contents = get_data(file)
	num_voxels = len(g_pred_contents)
	end = time.time()
	print("get subbatch data memmap: " + str(end-start))
	### GET PREDICTIONS ABOVE ###

	final_rankings = []

	spotlight_file_path = "/n/shieber_lab/Lab/users/cjou/true_spotlights_memmap/"
	print("iterating through file...")
	for pred_index in [1]: #tqdm(range(num_voxels)):
		if args.model_to_brain:
			g_true_activations = get_true_activations(args, spotlight_file_path, g_file_name, pred_index)
			rank = compare_rankings_to_brain(pred_index, g_true_activations)
			final_rankings.append(rank)

	to_save_file = "/n/shieber_lab/Lab/users/cjou/rankings_od32/batch-rankings-" + file_name + "-" + str(g_args.batch_num) + "of" + str(g_args.total_batches) + ".p"
	gc.disable()
	with open(to_save_file, "wb") as f:
		pickle.dump(final_rankings, f)
		gc.enable()
	return 

def main():
	global g_args
	global g_file_name

	argparser = argparse.ArgumentParser(description="calculate rankings for model-to-brain")
	argparser.add_argument("-embedding_layer", "--embedding_layer", type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("-batch_num", "--batch_num", type=int, help="batch number of total (for scripting) (out of --total_batches)", required=True)
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
	g_args = argparser.parse_args()

	# check conditions // can remove when making pipeline
	if g_args.brain_to_model and g_args.model_to_brain:
		print("select only one flag for brain_to_model or model_to_brain")
		exit()
	if not g_args.brain_to_model and not g_args.model_to_brain:
		print("select at least flag for brain_to_model or model_to_brain")
		exit()

	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(g_args)

	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/rankings_od32/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/rankings_od32/')

	embed_matrix = []

	specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"	
	g_file_name = specific_file.format(
		g_args.subject_number, 
		g_args.language, 
		g_args.num_layers, 
		g_args.model_type, 
		g_args.which_layer, 
		g_args.agg_type
	)

	print("\nNEW RUN")
	print("calculating average rank...")
	start = time.time()
	calculate_average_rank(g_args, g_file_name, embed_matrix)
	end = time.time()
	print("time: " + str(end-start)) 
	print("done.")

if __name__ == "__main__":
	main()