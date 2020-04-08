from tqdm import tqdm
import scipy.io
import pickle
import numpy as np
import sys
import argparse
import os
import helper

def get_file(args, file_name):
	path = "../mat_original/"
	if args.ranking:
		metric = "ranking"	
	elif args.rmse:
		metric = "rmse"
	elif args.llh:
		metric = "llh"
	elif args.fdr:
		metric = "fdr"
		path = "../fdr/"
	else:
		print("error: check for valid method of correlation")
		path = ""
	save_path = path + str(file_name) + "-3dtransform-" + str(metric)

	# print("LOADING FILE: " + str(save_path) + ".mat")
	values = scipy.io.loadmat(save_path + ".mat")

	if args.fdr:
		pvals = scipy.io.loadmat(save_path + "-pvals.mat")
	else:
		pvals = []
	return values["metric"], pvals

def generate_file_name(args, subject_number, which_layer):
	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

	if args.bert or args.word2vec or args.glove:
		specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(
			bertlabel) + str(direction) + str(validate) + "-subj{}-{}_layer{}"
		file_name = specific_file.format(
			subject_number,
			args.agg_type,
			which_layer
		)
	else:
		specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(
			bertlabel) + str(direction) + str(
			validate) + "-subj{}-parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
		file_name = specific_file.format(
			subject_number,
			args.language,
			args.num_layers,
			"brnn",
			which_layer,
			args.agg_type
		)
	return file_name

def main():
	argparser = argparse.ArgumentParser(description="layer and subject group level comparison")
	argparser.add_argument("-subject_number", "--subject_number", type=int, default=1,
						   help="subject number (fMRI data) for decoding")

	### SPECIFY MODEL PARAMETERS ###
	argparser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation",
						   action='store_true', default=False)
	argparser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model",
						   action='store_true', default=False)
	argparser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain",
						   action='store_true', default=False)
	argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str,
						   default='avg')
	argparser.add_argument("-language", "--language",
						   help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str,
						   default='spanish')
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, required=True)
	argparser.add_argument("-random", "--random", action='store_true', default=False,
						   help="True if initialize random brain activations, False if not")
	argparser.add_argument("-rand_embed", "--rand_embed", action='store_true', default=False,
						   help="True if initialize random embeddings, False if not")
	argparser.add_argument("-glove", "--glove", action='store_true', default=False,
						   help="True if initialize glove embeddings, False if not")
	argparser.add_argument("-word2vec", "--word2vec", action='store_true', default=False,
						   help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("-bert", "--bert", action='store_true', default=False,
						   help="True if initialize bert embeddings, False if not")
	argparser.add_argument("-normalize", "--normalize", action='store_true', default=False,
						   help="True if add normalization across voxels, False if not")
	argparser.add_argument("-permutation", "--permutation", action='store_true', default=False,
						   help="True if permutation, False if not")
	argparser.add_argument("-permutation_region", "--permutation_region", action='store_true', default=False,
						   help="True if permutation by brain region, False if not")

	### PLOTTING ###
	argparser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]",
						   type=int, default=1)

	### SPECIFY FOR SINGLE SUBJECT OR GROUP LEVEL ANALYSIS ###
	argparser.add_argument("-single_subject", "--single_subject", help="if single subject analysis",
						   action='store_true', default=False)
	argparser.add_argument("-group_level", "--group_level", help="if group level analysis", action='store_true',
						   default=False)
	argparser.add_argument("-searchlight", "--searchlight", help="if searchlight", action='store_true', default=False)
	
	### SPECIFY FOR ONE LAYER OR DIFFERENCE IN LAYERS ###
	argparser.add_argument("-single_layer", "--single_layer", help="if single layer significance",
						   action='store_true', default=False)
	argparser.add_argument("-across_layer", "--across_layer", help="if across layer depth significance",
						   action='store_true', default=False)

	### SPECIFY WHICH METRIC ### 
	argparser.add_argument("-fdr", "--fdr", help="if apply FDR", action='store_true', default=False)
	argparser.add_argument("-llh", "--llh", action='store_true', default=False,
						   help="True if calculate likelihood, False if not")
	argparser.add_argument("-ranking", "--ranking", action='store_true', default=False,
						   help="True if calculate ranking, False if not")
	argparser.add_argument("-rmse", "--rmse", action='store_true', default=False,
						   help="True if calculate rmse, False if not")
	argparser.add_argument("-rsa", "--rsa", action='store_true', default=False,
						   help="True if calculate rsa, False if not")

	argparser.add_argument("-local",  "--local", action='store_true', default=False, help="True if running locally")
	argparser.add_argument("-save_by_voxel",  "--save_by_voxel", action='store_true', default=False, help="True if save by voxel")

	args = argparser.parse_args()

	if args.num_layers != 12 and args.bert:
		print("error: please ensure bert has 12 layers")
		exit()

	if args.num_layers != 1 and (args.word2vec or args.random or args.permutation or args.glove):
		print("error: please ensure baseline has 1 layerc")
		exit()

	if not args.fdr and not args.llh and not args.ranking and not args.rmse:
		print("error: select at least 1 metric of correlation")
		exit()

	print("NUMBER OF LAYERS: " + str(args.num_layers))
	subjects = [1,2,4,5,7,8,9,10,11]

	print("getting common brain space")
	common_space = helper.load_common_space(subjects, local=args.local)
	voxel_coordinates = np.transpose(np.nonzero(common_space))
	print(voxel_coordinates.shape)
	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)
	# print("generating file names...")
	# layer1_file_name = generate_file_name(args, args.layer1)
	# layer2_file_name = generate_file_name(args, args.layer2)

	# print("retrieving file contents...")
	# layer1 = get_file(args, layer1_file_name)
	# layer2 = get_file(args, layer2_file_name)

	# print("evaluating layers...")
	# diff = compare_layers(layer1, layer2)
	# print("DIFF")
	# print(np.sum(diff))
	if args.fdr:
		metric = "fdr"
	if args.rmse:
		metric = "rmse"
	if args.rsa:
		metric = "rsa"
	if args.ranking:
		metric = "ranking"
	if args.llh:
		metric = "llh"

	# generate heatmap
	if args.single_subject and args.across_layer:
		first = True
		for layer_num in list(range(1, args.num_layers + 1)):
			print("generating file names...")
			layer_file_name = generate_file_name(args, args.subject_number, layer_num)

			print("retrieving file contents...")
			layer, pvals = get_file(args, layer_file_name)

			if first:
				updated_brain = layer
				best_layer = layer.astype(bool) * layer_num
				first = False
				mask = layer.astype(bool)
			else:
				if args.llh or args.ranking:
					max_vals = np.maximum(updated_brain, layer)
					# temp = np.minimum(updated_brain, layer)
					# print("SAME: " + str(np.sum(np.equal(max_vals, temp).astype(bool) * mask)))
				elif args.rmse:
					max_vals = np.minimum(updated_brain, layer)
				else:
					print("select llh, ranking, or rmse")
					exit()

				from_layer = np.equal(max_vals, layer).astype(bool) * mask * layer_num
				temp_best_layer = np.equal(max_vals, updated_brain).astype(bool) * mask * best_layer
				best_layer = np.maximum(from_layer, temp_best_layer)
				# print("NEW: " + str(np.sum(from_layer.astype(bool))))
				# print("OLD: " + str(np.sum(temp_best_layer.astype(bool))))
				updated_brain = max_vals

		if args.bert:
			file_name = "bert{}{}subj{}_{}".format(
						direction,
						validate,
						args.subject_number,
						args.agg_type,
					)
		else:
			specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-parallel-english-to-{}-model-{}layer-{}-pred-layer-{}"
			file_name = specific_file.format(
				args.subject_number,
				args.language,
				args.num_layers,
				"brnn",
				args.agg_type
			)

		print("BEST LAYER")
		total = 0 
		for layer in range(1, num_layers +1):
			print("LAYER" + str(layer))
			print(np.sum(best_layer == layer))
			total += np.sum(best_layer == layer)
		print("TOTAL:" + str(total))

		scipy.io.savemat("../" + str(file_name) + "_best_" + str(metric) + ".mat", dict(metric = best_layer.astype(np.int16)))

	if args.save_by_voxel:
		
		per_layer = []
		for layer_num in tqdm(list(range(1, args.num_layers + 1))):
			per_subject = []
			for subj_num in subjects:
				layer_file_name = generate_file_name(args, args.subject_number, layer_num)

				# print("retrieving file contents...")
				layer, _ = get_file(args, layer_file_name)
				voxel_values = layer[np.nonzero(common_space)]
				# print("LENGTH: " + str(len(voxel_values)))
				per_subject.append(voxel_values)
			per_layer.append(0.5 * np.transpose(np.array(per_subject)))
		
		per_voxel = np.stack( per_layer, axis=-1 )
		print(per_voxel.shape)
		print(per_voxel[0].shape)
		print(per_voxel[0])
		scipy.io.savemat("../mfit/bert_best_" + str(metric) + "_by_voxel.mat", dict(metric = per_voxel.astype(np.float32)))

	print("done.")
	return

if __name__ == "__main__":
	main()