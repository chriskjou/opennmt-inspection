import numpy as np
import argparse
from tqdm import tqdm
import pickle
from odyssey_decoding import normalize_voxels

def calculate_euclidean_distance(a, b):
	return np.sqrt(np.sum((a-b)**2))

def get_file_name(file_path, specific_file, args, i):
	file_name = specific_file + "_residuals_part" + str(i) + "of" + str(args.total_batches) + "-decoding-predictions.p"
	print("FILE NAME: " + str(file_path + file_name))
	return file_path + file_name

def get_voxel_number(batch_size, VOXEL_NUMBER, i):
	return batch_size * VOXEL_NUMBER + i

def set_voxel_number(file_path, specific_file, args, i):
	file_name = specific_file + "_residuals_part" + str(i) + "of" + str(args.total_batches) + "-decoding-predictions.p"
	file_contents = pickle.load( open( file_path + file_name, "rb" ) )
	return len(file_contents)

def compare_rankings_to_brain(predictions, volmask, modified_activations, which_match_point, VOXEL_NUMBER):
	a,b,c = volmask.shape
	nonzero_pts = np.transpose(np.nonzero(volmask))
	distances = []

	for pt_index in range(len(nonzero_pts)):
		# SPHERE MASK BELOW
		sphere_mask = np.zeros((a,b,c))
		x1,y1,z1 = nonzero_pts[pt_index]
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

		if len(spotlights) == len(predictions):
			dist = calculate_euclidean_distance(spotlights, predictions)
			distances.append(dist)
	
	distances.sort()
	index_of_dist = np.where(distances == which_match_point)
	voxel_number = get_voxel_number(which_file1, VOXEL_NUMBER, which_match_point)
	return {voxel_number: index_of_dist}

def compare_rankings_to_embeddings(prediction, embeddings):
	return

def calculate_voxel_distance_given_batch_files(first_file_contents, which_file1, VOXEL_NUMBER):
	voxel_rankings = {}

	num_voxels_file1 = len(first_file_contents)
	num_voxels_file2 = len(second_file_contents)
	total_voxels = len(first_file_contents) + len(second_file_contents)

	print("calculating across two files...")
	for i in range(total_voxels):
		for j in range(i + 1, total_voxels):
			print("I, J: " + str(i) + ", " + str(j))
			if i < num_voxels_file1 and j < num_voxels_file1:
				if not first_round_completed and first_file_contents[i][0].shape == first_file_contents[j][0].shape:
					dist = calculate_euclidean_distance(first_file_contents[i][0], first_file_contents[j][0])
					voxel_a = get_voxel_number(which_file1, VOXEL_NUMBER, i)
					voxel_b = get_voxel_number(which_file1, VOXEL_NUMBER, j)
					voxel_rankings[(voxel_a,voxel_b)] = dist
			elif i < num_voxels_file1 and j>= num_voxels_file1:
				modified_index = j - num_voxels_file1
				if first_file_contents[i][0].shape == second_file_contents[modified_index][0].shape:
					dist = calculate_euclidean_distance(first_file_contents[i][0], second_file_contents[modified_index][0])
					voxel_a = get_voxel_number(which_file1, VOXEL_NUMBER, i)
					voxel_b = get_voxel_number(which_file2, VOXEL_NUMBER, modified_index)
					voxel_rankings[(voxel_a,voxel_b)] = dist
			else: # i >= num_voxels_file1 and j>= num_voxels_file1:
				modified_index1 = i - num_voxels_file1
				modified_index2 = j - num_voxels_file2
				if not previously_calculated and second_file_contents[modified_index1][0].shape == second_file_contents[modified_index2][0].shape:
					dist = calculate_euclidean_distance(second_file_contents[modified_index1][0], second_file_contents[modified_index2][0])
					voxel_a = get_voxel_number(which_file2, VOXEL_NUMBER, modified_index1)
					voxel_b = get_voxel_number(which_file2, VOXEL_NUMBER, modified_index2)
					voxel_rankings[(voxel_a,voxel_b)] = dist
	return voxel_rankings

def calculate_average_rank(file_path, specific_file, args, volmask, modified_activations, embeddings, match_points_file):
	final_rankings = {}
	VOXEL_NUMBER = set_voxel_number(file_path, specific_file, args, 0)

	print("across all batches...")
	for i in tqdm(range(args.total_batches)):
		file = get_file_name(file_path, specific_file, args, i)
		file_contents = pickle.load( open( file, "rb" ) )
		num_voxels = len(file_contents)

		for pred_index in range(num_voxels):
			# if args.brain_to_model:
			# 	prediction = embeddings
			# 	voxel_dict = compare_rankings_to_embeddings(file_contents, embeddings)
			if args.model_to_brain:
				match_points_file_path = match_points_file.format(i)
				match_points = 	pickle.load( open(match_points_file_path +"-match-points.p", "rb" ) )
				voxel_dict = compare_rankings_to_brain(file_contents[pred_index], volmask, modified_activations, match_points[pred_index], VOXEL_NUMBER)
		
		final_rankings.update(voxel_dict)
	return final_rankings

def main():
	argparser = argparse.ArgumentParser(description="concatenate residuals/predictions from the relevant batches")
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
	args = argparser.parse_args()

	# check conditions // can remove when making pipeline
	if args.brain_to_model and args.model_to_brain:
		print("select only one flag for brain_to_model or model_to_brain")
		exit()
	if not args.brain_to_model and not args.model_to_brain:
		print("select at least flag for brain_to_model or model_to_brain")
		exit()

	if args.brain_to_model:
		direction = "brain2model_"
	else:
		direction = "model2brain_"

	if args.cross_validation:
		validate = "cv_"
	else:
		validate = "nocv_"

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

	if args.permutation:
		plabel = "permutation_"
	else:
		plabel = ""

	if args.permutation_region:
		prlabel = "permutation_region_"
	else:
		prlabel = ""

	### PREDICTIONS BELOW ###
	file_path = "/n/shieber_lab/Lab/users/cjou/predictions/"
	specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"	
	file_name = specific_file.format(
		args.subject_number, 
		args.language, 
		args.num_layers, 
		args.model_type, 
		args.which_layer, 
		args.agg_type
	)
	### PREDICTIONS ABOVE ###

	### EMBEDDINGS BELOW ###

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
			embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/word2vec/" + str(file_name) + ".p", "rb" ) )	
		elif args.glove:
			# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
			embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/glove/" + str(file_name) + ".p", "rb" ) )	
		elif args.bert:
			# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
			embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/bert/" + str(file_name) + ".p", "rb" ) )
		else: # args.rand_embed
			# embed_matrix = pickle.load( open( "../embeddings/glove/" + str(file_name) + ".p", "rb" ) )
			embed_matrix = pickle.load( open( "/n/shieber_lab/Lab/users/cjou/embeddings/rand_embed/rand_embed.p", "rb" ) )	
	### EMBEDDINGS ABOVE ###

	### BRAIN ACTIVATIONS BELOW ###
	volmask = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{subj_num}/volmask.p", "rb" ) )
	modified_activations = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{subj_num}/" + str(plabel) + str(prlabel) + "modified_activations.p", "rb" ) )

	if args.normalize:
		modified_activations = normalize_voxels(modified_activations)

	if args.random:
		print("RANDOM ACTIVATIONS")
		modified_activations = np.random.randint(-20, high=20, size=(240, 79, 95, 68))
	### BRAIN ACTIVATIONS ABOVE ###

	temp_file_name = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj" + str(args.subject_number) + "-" + str(file_name) + "_residuals_part{}of" + str(total_batches)
	match_points_file = "/n/shieber_lab/Lab/users/cjou/match_points/" + temp_file_name
	print("MATCH POINTS FILE: " + str(match_points_file))

	print("calculating average rank...")
	final_rankings = calculate_average_rank(file_path, file_name, args, volmask, modified_activations, match_points_file)
	pickle.dump( final_rankings, open("/n/shieber_lab/Lab/users/cjou/final_predictions/concatenated-" + file_name + ".p", "wb" ) )
	print("done.")

if __name__ == "__main__":
	main()