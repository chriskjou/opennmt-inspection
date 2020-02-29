import numpy as np
import pickle
import sys
import argparse
import os
import helper

def concatenate_all(specific_file, args, type_concat):
	final_residuals = []
	for i in range(args.total_batches):
		file_path = "/n/shieber_lab/Lab/users/cjou/residuals_od32/"
		# file_path = "../residuals/"
		# specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(glabel) + str(w2vlabel) + str(direction) + str(validate) + "-subj" + str(args.subject_number) + "-parallel-english-to-" + str(args.language) + "-model-" + str(args.num_layers) + "layer-" + str(args.model_type) + "-pred-layer" + str(args.which_layer) + "-" + str(args.agg_type)
		### specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj" + str(subject_number) + "-" + str(agg_type)
		### specific_file = "parallel-english-to-" + str(language) + "-model-" + str(num_layers) + "layer-" + str(model_type) + "-pred-layer" + str(layer) + "-" + str(agg_type)
		### if type_concat == 'residuals':
			### file_name = "../residuals/" + specific_file + "_residuals_part" + str(i) + "of" + str(total_batches) + ".p"
		file_name = specific_file + "_residuals_part" + str(i) + "of" + str(args.total_batches)
		if type_concat == 'predictions':
			file_path = "../predictions/"
			# file_path = "/n/shieber_lab/Lab/users/cjou/predictions/"
			file_name += "-decoding-predictions"
		# print("FILE NAME: " + str(file_name))
		print("FILE NAME: " + str( file_path + file_name + ".p"))
		part = pickle.load( open( file_path + file_name + ".p", "rb" ) )
		final_residuals.extend(part)
	return final_residuals

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

	# languages = 'spanish' #['spanish', 'german', 'italian', 'french', 'swedish']
	# num_layers = 2 #[2, 4]
	# model_type = 'brnn' #['brnn', 'rnn']
	# agg_type = ['avg', 'max', 'min', 'last']
	# subj_num = 1
	# nbatches = 100

	# check conditions // can remove when making pipeline
	if args.brain_to_model and args.model_to_brain:
		print("select only one flag for brain_to_model or model_to_brain")
		exit()
	if not args.brain_to_model and not args.model_to_brain:
		print("select at least flag for brain_to_model or model_to_brain")
		exit()

	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

	print("CROSS VALIDATION: " + str(args.cross_validation))
	print("BRAIN_TO_MODEL: " + str(args.brain_to_model))
	print("MODEL_TO_BRAIN: " + str(args.model_to_brain))
	print("GLOVE: " + str(args.glove))
	print("WORD2VEC: " + str(args.word2vec))
	print("BERT: " + str(args.bert))
	print("RANDOM BRAIN: " + str(args.random))
	print("RANDOM EMBEDDINGS: " + str(args.rand_embed))
	print("PERMUTATION: " + str(args.permutation))
	print("PERMUTATION REGION: " + str(args.permutation_region))

	#residual_name = args.residual_name
	#total_batches = args.total_batches

	# make final path
	if not os.path.isdir('/n/shieber_lab/Lab/users/cjou/rmses/'):
		os.mkdir('/n/shieber_lab/Lab/users/cjou/rmses/')

	if not os.path.isdir('/n/shieber_lab/Lab/users/cjou/final_predictions/'):
		os.mkdir('/n/shieber_lab/Lab/users/cjou/final_predictions/')

	for atype in [args.agg_type]:
		# for layer in list(range(1, num_layers+1)):
		for layer in [args.which_layer]:
			print("LAYER: " + str(layer))
			if not args.word2vec and not args.glove and not args.bert and not args.random:
				specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
				file_format = specific_file.format(
					args.subject_number, 
					args.language, 
					args.num_layers, 
					args.model_type, 
					args.which_layer, 
					args.agg_type
				)
			else:
				file_format = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-{}_layer{}".format(args.subject_number, args.agg_type, args.which_layer)

			final_residuals = concatenate_all(file_format, args, 'residuals')
			# final_predictions = concatenate_all(plabel, prlabel, rlabel, elabel, glabel, w2vlabel, bertlabel, args, direction, validate, 'predictions')
			
			# RMSES
			# specific_file = "parallel-english-to-" + str(args.language) + "-model-" + str(args.num_layers) + "layer-" + str(args.model_type) + "-pred-layer" + str(layer) + "-" + str(args.agg_type)
			
			file_path = "/n/shieber_lab/Lab/users/cjou/rmses/concatenated-"
			# file_path = "../rmses/concatenated-"
			file_name = file_path + str(file_format) + ".p"
			pickle.dump( final_residuals, open( file_name, "wb" ) )

			# # PREDICTIONS
			# file_name = "/n/shieber_lab/Lab/users/cjou/final_predictions/concatenated-" + str(file_format) + ".p"
			# pickle.dump( final_predictions, open( file_name, "wb" ) )
	print("done.")
	return

if __name__ == "__main__":
	main()
