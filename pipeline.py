import sys
import subprocess
import os
from tqdm import tqdm

def main():

	############# GET ARGUMENTS #############
	parser = argparse.ArgumentParser()

	parser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	parser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	parser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	parser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	parser.add_argument("-subj_num", "--subj_num", help="fMRI subject number ([1:11])", type=int, default=1)
	parser.add_argument("-nbatches", "--nbatches", help="Total number of batches to run", type=int, default=100)
	parser.add_argument("-create_model", "--create_model", help="Create OpenNMT prediction model", type=bool, default='False')
	parser.add_argument("-format_data", "--format_data", help="Format fMRI data", type=bool, default='False')

	args = parser.parse_args()

	languages = ['spanish', 'german', 'italian', 'french', 'swedish']
	num_layers = [2, 4]
	model_type = ['brnn', 'rnn']
	agg_type = ['avg', 'max', 'min', 'last']
	subj_num = range(1,12)
	nbatches = 100

	############# VALIDATE ARGUMENTS #############
	if args.language not in languages:
		print("invalid language")
		exit()
	if args.num_layers not in num_layers:
		print("invalid num_layer")
		exit()
	if args.model_type not in model_type:
		print("invalid model_type")
		exit()
	if args.agg_type not in agg_type:
		print("invalid agg_type")
		exit()
	if args.subj_num not in subj_num:
		print("invalid subj_num")
		exit()
	if args.which_layer not in list(range(args.num_layers)):
		print("invalid which_layer: which_layer must be between 1 and args.num_layers, inclusive")
		exit()

	############# CREATE MODEL #############
	
	if args.create_model:
		### multiparallelize texts
		### todo: add here

		### preprocess ### train ### translate
		training_src = ""
		training_tgt = ""
		validation_src = ""
		validation_tgt = ""
		preprocess = "python preprocess.py -train_src ../multiparallelize/training/" + str(training_text) + 
						" -train_tgt ../multiparallelize/training/" + str(training_tgt) + 
						" -valid_src ../multiparallelize/validation/" + str(validation_src) + 
						" -valid_tgt ../multiparallelize/training/validation/" + str(validation_tgt) + 
						" -save ../multiparallelize"
		os.system(preprocess)
		train = "python train.py -data data/english-to-spanish -save_model small-english-to-spanish-model -gpu 0 -separate_layers"
		os.system(train)
		translate = "python translate.py -model ../final_models/english-to-spanish-model_acc_61.26_ppl_6.28_e13.pt -src cleaned_sentencesGLM.txt -output ../predictions/english-to-spanish-model-pred.txt -replace_unk -verbose -dump_layers ../predictions/english-to-spanish-model-pred.pt"
		os.system(translate)

	############# FORMAT DATA #############
	if args.format_data:
		format_cmd = "python format_for_subject.py --subject_number " + str(subj_num)
		os.system(format_cmd)

	############# DECODING #############
	# usage: python make_scripts.py -language -num_layers -type -which_layer -agg_type -subj_num -num_batches"
	cmd = "python make_scripts.py"
	options = "--language " + str(args.language) + 
				" --num_layers " + str(args.num_layers) + 
				" --type " + str(args.type) + 
				" --which_layer " + str(args.which_layer) + 
				" --agg_type " + str(args.agg_type) + 
				" --subj_num " + str(args.subj_num) + 
				" --num_batches" + str(args.num_batches)
	entire_cmd = cmd + " " + options
	os.system(entire_cmd)

	### wait for job dependency

	# combine RMSE
	# usage: python get_residuals.py --residual_name XXXXX --total_batches X
	rmse = "python get_residuals.py" 
	options = "--residual_name " + str(ADD HERE) + 
				" --total_batches 100"
	entire_cmd = rmse + " " + options
	os.system(entire_command)

	return

if __name__ == "__main__":
    main()