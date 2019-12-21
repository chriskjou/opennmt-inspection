import numpy as np
import argparse
from tqdm import tqdm

def calculate_euclidean_distance(a, b):
	return np.sqrt(np.sum((a-b)**2))

def get_file_name(file_path, specific_file, args, i):
	file_name = specific_file + "_residuals_part" + str(i) + "of" + str(args.total_batches) + "-decoding-predictions.p"
	print("FILE NAME: " + str( file_path + file_name))
	return file_path + file_name

def get_voxel_number(batch_size, VOXEL_NUMBER, i):
	return batch_size * VOXEL_NUMBER + i

def set_voxel_number(file_path, specific_file, args, i):
	file_name = specific_file + "_residuals_part" + str(i) + "of" + str(args.total_batches) + "-decoding-predictions.p"
	file_contents = pickle.load( open( file_name, "rb" ) )
	return len(file_contents)

def calculate_voxel_distance_given_batch_files(first_file_contents, second_file_contents, which_file1, which_file2, VOXEL_NUMBER, first_round_completed, previously_calculated):
	voxel_rankings = {}

	num_voxels_file1 = len(first_file_contents)
	num_voxels_file2 = len(second_file_contents)
	total_voxels = len(first_file_contents) + len(second_file_contents)

	print("calculating across two files...")
	for i in tqdm(range(total_voxels)):
		for j in range(i + 1, total_voxels):
			if i < num_voxels_file1 and j < num_voxels_file1:
				if not first_round_completed:
					dist = calculate_euclidean_distance(first_file_contents[i][0], first_file_contents[j][0])
					voxel_a = get_voxel_number(which_file1, VOXEL_NUMBER, i)
					voxel_b = get_voxel_number(which_file1, VOXEL_NUMBER, j)
					voxel_rankings[(voxel_a,voxel_b)] = dist
			elif i < num_voxels_file1 and j>= num_voxels_file1:
				dist = calculate_euclidean_distance(first_file_contents[i][0], second_file_contents[j][0])
				voxel_a = get_voxel_number(which_file1, VOXEL_NUMBER, i)
				voxel_b = get_voxel_number(which_file2, VOXEL_NUMBER, j)
				voxel_rankings[(voxel_a,voxel_b)] = dist
			else: # i >= num_voxels_file1 and j>= num_voxels_file1:
				if not previously_calculated:
					dist = calculate_euclidean_distance(second_file_contents[i][0], second_file_contents[j][0])
					voxel_a = get_voxel_number(which_file2, VOXEL_NUMBER, i)
					voxel_b = get_voxel_number(which_file2, VOXEL_NUMBER, j)
					voxel_rankings[(voxel_a,voxel_b)] = dist
	return voxel_rankings

def calculate_average_rank(file_path, specific_file, args):
	final_rankings = {}
	VOXEL_NUMBER = set_voxel_number(file_path, specific_file, args, 1)
	first_round_completed = False
	previously_calculated = False
	print("across all batches...")
	for i in tqdm(range(args.total_batches)):
		for j in list(range(i+1, args.total_batches)):
			first_file = get_file_name(file_path, specific_file, args, i)
			second_file = get_file_name(file_path, specific_file, args, j)
			
			first_file_contents = pickle.load( open( first_file, "rb" ) )
			second_file_contents = pickle.load( open( second_file, "rb" ) )
			
			voxel_dict = calculate_voxel_distance_given_batch_files(first_file_contents, second_file_contents, i, j, VOXEL_NUMBER, first_round_completed, previously_calculated)

			if not first_round_completed:
				first_round_completed = True

			final_rankings.update(voxel_dict)
		previously_calculated = True
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

	print("calculating average rank...")
	final_rankings = calculate_average_rank(file_path, file_name, args)
	pickle.dump( final_rankings, open("/n/shieber_lab/Lab/users/cjou/final_predictions/concatenated-" + file_name + ".p", "wb" ) )
	print("done.")

if __name__ == "__main__":
	main()