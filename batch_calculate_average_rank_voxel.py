import numpy as np
import argparse
from tqdm import tqdm
import pickle
import scipy.io
import os
import math
import time
from numba import jit, cuda 
import multiprocessing as mp
import gc

spotlight_file_path = "/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/"

def get_embed_matrix(embedding):
	dict_keys = list(embedding.keys())[3:]
	embed_matrix = np.array([embedding[i][0][1:] for i in dict_keys])
	in_training_bools = np.array([embedding[i][0][0] for i in dict_keys])
	return embed_matrix

# @jit(nopython=True, parallel=True)
# def calculate_euclidean_distance(a, b):
# 	return np.sqrt(np.sum((a-b)**2))

@cuda.jit
def calculate_euclidean_distance(a, b, dist):
	x,y = a.shape
	running_sum = 0

	for i in range(x):
		for j in range(y):
			running_sum += ((b[i][j] - a[i][j])**2)

	dist = math.sqrt(running_sum)

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
	gc.disable()
	with open(file_path + file_name, "rb") as f:
		file_contents = pickle.load(f)
		gc.enable()
		return len(file_contents)

def get_true_activations(args, file_path, file_name, pred_index):
	entire_file_name = file_name + "_residuals_part" + str(args.batch_num) + "of" + str(args.total_batches) + "-true-spotlights.p"
	gc.disable()
	with open(file_path + entire_file_name, "rb") as f:
		file_contents = pickle.load(f)
		gc.enable()
		return file_contents[pred_index]

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

def compare_rankings_to_brain(args, file_name, predictions, true_activations, VOXEL_NUMBER, radius=5):

	### GET ALL SPOTLIGHT BRAIN ACTIVATIONS BELOW ###
	file_path = "/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/"
	### GET ALL SPOTLIGHT BRAIN ACTIVATIONS ABOVE ###

	# distances = []
	predicted_distance = np.ones(1, dtype=np.float32)
	calculate_euclidean_distance(np.array(predictions), np.array(true_activations), predicted_distance)
	# predicted_distance = calculate_euclidean_distance(np.array(predictions), np.array(true_activations))

	rank = 0

	batches = chunkify(range(args.total_batches), args.sub_batch_num, args.total_sub_batches)

	for i in batches:
		# print("BATCH: " + str(i))

		spotlight_activations_in_batch_file_name = get_file_name(args, file_path, file_name, i, true_activations=True)
		gc.disable()
		with open(spotlight_activations_in_batch_file_name, "rb") as f:
			spotlight_activations = pickle.load(f)
			gc.enable()

			# iterate over each sentence
			for sentence_act in spotlight_activations:
				if np.array_equal(np.array(predictions).shape, np.array(sentence_act).shape):
					dist = np.ones(1, dtype=np.float32)
					calculate_euclidean_distance(np.array(predictions), np.array(sentence_act), dist)
					# dist = calculate_euclidean_distance(np.array(predictions), np.array(sentence_act))
					# print("DISTANCE: " + str(dist))
					# print("PREDICTED: " + str(predicted_distance))
					if dist <= predicted_distance:
						rank+=1
					# distances.append(dist)

			# REMOVE FROM MEMORY
			del spotlight_activations

	# CALCULATE PREDICTED DISTANCE
	# distances = np.array(distances)
	
	# rank = len(np.where(distances < predicted_distance))

	# FIND RANKING OF PREDICTED DISTANCE
	# voxel_number = get_voxel_number(which_file, VOXEL_NUMBER, index_within_activations)

	return rank

def compare_rankings_to_embeddings(prediction, embeddings):
	return

def multiparallelize_voxel(pred_index, pred_contents):
	global spotlight_file_path
	global g_file_name
	global g_args

	print("PRED INDEX: " + str(pred_index))
	true_activations = get_true_activations(g_args, spotlight_file_path, g_file_name, pred_index)
	rank = compare_rankings_to_brain(g_args, g_file_name, pred_contents, true_activations, 0) # REMOVE VOXEL NUMBER
	del true_activations
	return rank
	# return pred_index

def calculate_average_rank(args, file_name, embeddings):
	global g_args

	### PREDICTIONS BELOW ###
	file_path = "/n/shieber_lab/Lab/users/cjou/predictions_od32/"
	### PREDICTIONS ABOVE ###

	### GET PREDICTION FILE BELOW ###
	file = get_file_name(g_args, file_path, g_file_name, g_args.batch_num)
	gc.disable()
	with open(file, "rb") as f:
		pred_contents = pickle.load(f)
		gc.enable()
		num_voxels = len(pred_contents)
	### GET PREDICTION FILE ABOVE ###

	### GET DICTIONARY MATCHING FILE BELOW ###
	# matches_file_path = "/n/shieber_lab/Lab/users/cjou/match_points/" 
	# temp_file_name = str(file_name) + "_residuals_part" + str(args.batch_num) + "of" + str(args.total_batches)
	# match_points_file = matches_file_path + temp_file_name
	# print("MATCH POINTS FILE: " + str(match_points_file))
	### GET DICTIONARY MATCHING FILE ABOVE ###

		### FIND VOXEL NUMBER OF BATCH ###
		VOXEL_NUMBER = set_voxel_number(g_args, file_path, g_file_name)
		### FIND VOXEL NUMBER OF BATCH ###

		final_rankings = []

		# match_points_file_path = match_points_file.format(i)
		# match_points = 	pickle.load( open(match_points_file_path +"-match-points.p", "rb" ) )

		# NUM_THREADS = 5
		# print("iterating through file...")
		# print("CPU COUNT: " + str(mp.cpu_count()))
		# print("NUM THREADS: " + str(NUM_THREADS))
		# pool = mp.Pool(NUM_THREADS)
		# final_rankings = [pool.apply(multiparallelize_voxel, args=(pred_index, pred_contents[pred_index])) for pred_index in range(num_voxels)]
		# pool.close()    
		for pred_index in tqdm(range(num_voxels)):
			if args.model_to_brain:
				true_activations = get_true_activations(args, spotlight_file_path, file_name, pred_index)
				rank = compare_rankings_to_brain(args, file_name, pred_contents[pred_index], true_activations, VOXEL_NUMBER)
				final_rankings.append(rank)
				del true_activations

		to_save_file = "/n/shieber_lab/Lab/users/cjou/rankings_od32/batch-rankings-" + file_name + "-" + str(g_args.batch_num) + "of" + str(g_args.total_batches) + "-subbatch" + str(g_args.sub_batch_num) + ".p"
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
	argparser.add_argument("-sub_batch_num", "--sub_batch_num", type=int, help="chunkify sub batch number to run euclidean distance", required=True)
	argparser.add_argument("-total_sub_batches", "--total_sub_batches", type=int, help="total number of sub_batches to run euclidean distance", required=True)
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

	if g_args.brain_to_model:
		direction = "brain2model_"
	else:
		direction = "model2brain_"

	if g_args.cross_validation:
		validate = "cv_"
	else:
		validate = "nocv_"

	if g_args.random:
		rlabel = "random"
	else:
		rlabel = ""

	if g_args.rand_embed:
		elabel = "rand_embed"
	else:
		elabel = ""
		
	if g_args.glove:
		glabel = "glove"
	else:
		glabel = ""

	if g_args.word2vec:
		w2vlabel = "word2vec"
	else:
		w2vlabel = ""

	if g_args.bert:
		bertlabel = "bert"
	else:
		bertlabel = ""

	if g_args.permutation:
		plabel = "permutation_"
	else:
		plabel = ""

	if g_args.permutation_region:
		prlabel = "permutation_region_"
	else:
		prlabel = ""

	if not os.path.exists('/n/shieber_lab/Lab/users/cjou/rankings/'):
		os.makedirs('/n/shieber_lab/Lab/users/cjou/rankings/')

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