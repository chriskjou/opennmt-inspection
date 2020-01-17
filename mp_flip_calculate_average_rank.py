import numpy as np
import argparse
from tqdm import tqdm
import pickle
import scipy.io
import os
import math
import time
# from numba import jit, cuda 
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

# @cuda.jit
# def calculate_euclidean_distance(a, b, dist):
# 	x,y = a.shape
# 	running_sum = 0

# 	for i in range(x):
# 		for j in range(y):
# 			running_sum += ((b[i][j] - a[i][j])**2)

# 	dist = math.sqrt(running_sum)

def get_file_name(args, file_path, specific_file, i, true_activations=False):
	file_name = specific_file + "_residuals_part" + str(i) + "of" + str(args.total_batches) 
	if not true_activations:
		# print("FILE NAME: " + str(file_path + file_name + "-decoding-predictions.p"))
		return file_path + file_name + "-decoding-predictions.p"
	# print("FILE NAME: " + str(file_path + file_name + "-true-spotlights.p"))
	return file_path + file_name + "-true-spotlights.p"

def get_voxel_number(batch_size, VOXEL_NUMBER, i):
	return batch_size * VOXEL_NUMBER + i

def set_voxel_number(args, file_path, file_name):
	file_name = file_name + "_residuals_part0of" + str(args.total_batches) + "-decoding-predictions.p"
	file_contents = pickle.load( open( file_path + file_name, "rb" ) )
	return len(file_contents)

def get_true_activations(args, file_path, file_name, pred_index):
	entire_file_name = file_name + "_residuals_part" + str(args.batch_num) + "of" + str(args.total_batches) + "-true-spotlights.p"
	file_contents = pickle.load(open(file_path + entire_file_name, "rb"))
	return file_contents[pred_index]

def mp_compare_rankings_to_brain(i, predictions):
	global g_args
	global g_file_name
	global g_predicted_distance

	### GET ALL SPOTLIGHT BRAIN ACTIVATIONS BELOW ###
	file_path = "/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/"
	### GET ALL SPOTLIGHT BRAIN ACTIVATIONS ABOVE ###

	spotlight_activations_in_batch_file_name = get_file_name(g_args, file_path, g_file_name, i, true_activations=True)
	rank = 0
	gc.disable()
	with open(spotlight_activations_in_batch_file_name, "rb") as f:
		spotlight_activations = pickle.load(f)
		gc.enable()

		# iterate over each sentence
		for sentence_act in spotlight_activations:
			if np.array_equal(np.array(predictions).shape, np.array(sentence_act).shape):
				# dist = np.ones(1, dtype=np.float32)
				# calculate_euclidean_distance(np.array(predictions), np.array(sentence_act), dist)
				dist = calculate_euclidean_distance(np.array(predictions), np.array(sentence_act))
				if dist <= g_predicted_distance:
					rank+=1
				# distances.append(dist)

		# REMOVE FROM MEMORY
		del spotlight_activations
	return rank

def compare_rankings_to_brain(predictions, true_activations, radius=5):
	global g_args
	global g_pred_contents
	global g_predicted_distance

	# distances = []
	# predicted_distance = np.ones(1, dtype=np.float32)
	# calculate_euclidean_distance(np.array(predictions), np.array(true_activations), predicted_distance)
	predicted_distance = calculate_euclidean_distance(np.array(predictions), np.array(true_activations))
	g_predicted_distance = predicted_distance

	# NUM_THREADS = 5
	# print("CUDA: ")
	# print(cuda.gpus)
	print("iterating through file...")
	print("CPU COUNT: " + str(mp.cpu_count()))
	print("CPU on slurm: " + str(int(os.environ["SLURM_CPUS_ON_NODE"])))
	# print("NUM THREADS: " + str(NUM_THREADS))

	pool = mp.Pool(processes=int(os.environ["SLURM_CPUS_ON_NODE"]))
	extra_arguments = [(pred_index, np.array(g_pred_contents[pred_index])) for pred_index in range(g_args.total_batches)]
	final_rankings = pool.starmap(mp_compare_rankings_to_brain, extra_arguments)
	pool.close()

	return np.sum(final_rankings)

def compare_rankings_to_embeddings(prediction, embeddings):
	return

def flip_mp_compare_rankings_to_brain(pred_index, true_index, radius=5):
	global g_args
	global g_file_name
	global g_predicted_distance

	### GET ALL SPOTLIGHT BRAIN ACTIVATIONS BELOW ###
	file_path = "/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/"
	### GET ALL SPOTLIGHT BRAIN ACTIVATIONS ABOVE ###

	true_spotlight_activations = get_file_name(g_args, file_path, g_file_name, true_index, true_activations=True)

	### PREDICTIONS BELOW ###
	file_path = "/n/shieber_lab/Lab/users/cjou/predictions_od32/"
	### PREDICTIONS ABOVE ###

	predicted_spotlight_activations = get_file_name(g_args, file_path, g_file_name, pred_index)

	distances = []

	gc.disable()
	with open(true_spotlight_activations, "rb") as f:
		spotlight_activations = pickle.load(f)

		with open(predicted_spotlight_activations, "rb") as f2:
			predicted_activations = pickle.load(f)
			gc.enable()

			# iterate over each sentence
			for sentence_act in spotlight_activations:
				for predicted_act in predicted_activations:
					if np.array_equal(np.array(predicted_act).shape, np.array(sentence_act).shape):
						# dist = np.ones(1, dtype=np.float32)
						# calculate_euclidean_distance(np.array(predictions), np.array(sentence_act), dist)
						dist = calculate_euclidean_distance(np.array(predicted_act), np.array(sentence_act))
						distances.append(dist)

			del predicted_activations
		del spotlight_activations
	return distances

def flip_calculate_average_rank(args, file_name, embeddings):
	global g_args
	global g_file_name
	global g_pred_contents
	global g_true_activations

	print("iterating through file...")
	print("CPU COUNT: " + str(mp.cpu_count()))
	print("CPU on slurm: " + str(int(os.environ["SLURM_CPUS_ON_NODE"])))
	# print("NUM THREADS: " + str(NUM_THREADS))

	pool = mp.Pool(processes=int(os.environ["SLURM_CPUS_ON_NODE"]))
	true_index = 0
	extra_arguments = [(pred_index, true_index) for pred_index in range(g_args.total_batches)]
	final_rankings = pool.starmap(flip_mp_compare_rankings_to_brain, extra_arguments)
	pool.close()
	return


def calculate_average_rank(args, file_name, embeddings):
	global g_args
	global g_file_name
	global g_pred_contents
	global g_true_activations

	### PREDICTIONS BELOW ###
	file_path = "/n/shieber_lab/Lab/users/cjou/predictions_od32/"
	### PREDICTIONS ABOVE ###

	### GET PREDICTION FILE BELOW ###
	file = get_file_name(g_args, file_path, g_file_name, args.batch_num)
	gc.disable()
	with open(file, "rb") as f:
		g_pred_contents = pickle.load(f)
		gc.enable()
		num_voxels = len(g_pred_contents)
	### GET PREDICTION FILE ABOVE ###

		### FIND VOXEL NUMBER OF BATCH ###
		# VOXEL_NUMBER = set_voxel_number(g_args, file_path, g_file_name)
		### FIND VOXEL NUMBER OF BATCH ###

		final_rankings = []

		spotlight_file_path = "/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/"
		print("iterating through file...")
		for pred_index in tqdm(range(num_voxels)):
			# if args.brain_to_model:
			# 	prediction = embeddings
			# 	voxel_dict = compare_rankings_to_embeddings(file_contents, embeddings)
			if args.model_to_brain:
				g_true_activations = get_true_activations(args, spotlight_file_path, g_file_name, pred_index)
				# print("WHICH MATCH POINT INDEX: " + str(match_points[pred_index]))
				# print(match_points[pred_index].shape)
				rank = compare_rankings_to_brain(g_pred_contents[pred_index], g_true_activations, 0)
		
				final_rankings.append(rank)
			# del voxel_dict
				# del g_true_activations

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

	### EMBEDDINGS BELOW ###
	# if not args.glove and not args.word2vec and not args.bert and not args.rand_embed:
	# 	embed_loc = args.embedding_layer
	# 	# file_name = embed_loc.split("/")[-1].split(".")[0]
	# 	embedding = scipy.io.loadmat(embed_loc)
	# 	embed_matrix = get_embed_matrix(embedding)
	# else:
	# 	embed_loc = args.embedding_layer
	# 	file_name = embed_loc.split("/")[-1].split(".")[0].split("-")[-1] # aggregation type
	# 	if args.word2vec:
	# 		# embed_matrix = pickle.load( open( "../embeddings/word2vec/" + str(file_name) + ".p", "rb" ) )	
	# 		embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/word2vec/" + str(file_name) + ".p", "rb" ) )	
	# 	elif args.glove:
	# 		# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
	# 		embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/glove/" + str(file_name) + ".p", "rb" ) )	
	# 	elif args.bert:
	# 		# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
	# 		embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/bert/" + str(file_name) + ".p", "rb" ) )
	# 	else: # args.rand_embed
	# 		# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
	# 		embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/rand_embed/rand_embed.p", "rb" ) )	
	### EMBEDDINGS ABOVE ###
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
	calculate_average_rank(g_args, g_file_name, embed_matrix)
	end = time.time()
	print("time: " + str(end-start)) 
	print("done.")

if __name__ == "__main__":
	main()