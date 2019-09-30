import numpy as np
import pickle
import sys
import argparse
import os

def concatenate_all_residuals(language, num_layers, model_type, layer, agg_type, total_batches):
	final_residuals = []
	for i in range(total_batches):
		specific_file = "parallel-english-to-" + str(language) + "-model-" + str(num_layers) + "layer-" + str(model_type) + "-pred-layer" + str(layer) + "-" + str(agg_type)
		file_name = "../residuals/" + specific_file + "_residuals_part" + str(i) + "of" + str(total_batches) + ".p"
		part = pickle.load( open( file_name, "rb" ) )
		final_residuals.extend(part)
	return final_residuals

def main():
	argparser = argparse.ArgumentParser(description="Concatenate residuals from the relevant batches")
	# argparser.add_argument("--residual_name", type=str, help="Stub of the residual name in /residuals " +
	#												"directory(spread over --total_batches from cluster)", required=True)
	argparser.add_argument("--total_batches", type=int, help="total number of batches "
														+ "residual_name is spread across", required=True)
	argparser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	argparser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	argparser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	argparser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	argparser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation", action='store_true', default=False)
	argparser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model", action='store_true', default=False)
	argparser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain", action='store_true', default=False)

	args = argparser.parse_args()
	languages = 'spanish' #['spanish', 'german', 'italian', 'french', 'swedish']
	num_layers = 2 #[2, 4]
	model_type = 'brnn' #['brnn', 'rnn']
	agg_type = ['avg', 'max', 'min', 'last']
	subj_num = 1
	nbatches = 100

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

	print(args.cross_validation)
	print(args.brain_to_model)
	print(args.model_to_brain)

	#residual_name = args.residual_name
	#total_batches = args.total_batches

	# make final path
	if not os.path.isdir('../rmses/'):
		os.mkdir('../rmses/')

	for atype in agg_type:
		for layer in list(range(1,num_layers+1)):
			final_residuals = concatenate_all_residuals(args.language, args.num_layers, args.model_type, layer, args.agg_type, args.total_batches)
			specific_file = str(direction) + str(validate) + "subj{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
			file_format = specific_file.format(
				args.subject_number, 
				args.language, 
				args.num_layers, 
				args.model_type, 
				args.which_layer, 
				args.agg_type
			)
			file_name = "../rmses/concatenated-" + str(file_format) + ".p"
			pickle.dump( final_residuals, open( file_name, "wb" ) )
	print("done.")
	return

if __name__ == "__main__":
	main()
