import sys
import subprocess
import os
from tqdm import tqdm

# global variables
languages = ['spanish', 'german', 'italian', 'french', 'swedish']
num_layers = [2, 4]
model_type = ['brnn', 'rnn']
agg_type = ['avg', 'max', 'min', 'last']
subj_num = range(1,12)
nbatches = 100

def main():

	############# GET ARGUMENTS #############
	parser = argparse.ArgumentParser(description="entire OpenNMT pipeline: data prep, model, decoding, visualization")
	parser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	parser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	parser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	parser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	parser.add_argument("-subj_num", "--subj_num", help="fMRI subject number ([1:11])", type=int, default=1)
	parser.add_argument("-nbatches", "--nbatches", help="Total number of batches to run", type=int, default=100)
	parser.add_argument("-create_model", "--create_model", help="Create OpenNMT prediction model", action='store_true', default=False)
	parser.add_argument("-format_data", "--format_data", help="Format fMRI data", action='store_true', default=False)
	parser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation", action='store_true', default=False)
	parser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model", action='store_true', default=False)
	parser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain", action='store_true', default=False)
	args = parser.parse_args()

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
	if args.brain_to_model and args.model_to_brain:
		print("select only one flag for brain_to_model or model_to_brain")
		exit()
	if not args.brain_to_model and not args.model_to_brain:
		print("select at least flag for brain_to_model or model_to_brain")
		exit()

	############# GLOBAL OPTIONS #############
	options = "--language " + str(args.language) + 
				" --num_layers " + str(args.num_layers) + 
				" --model_type " + str(args.type) + 
				" --which_layer " + str(args.which_layer) + 
				" --agg_type " + str(args.agg_type) + 
				" --subject_number " + str(args.subj_num)
	get_residuals_and_make_scripts = " --num_batches" + str(args.num_batches)
	if args.cross_validation:
		options += " --cross_validation"
	if args.brain_to_model:
		options += " --brain_to_model"
	if args.model_to_brain:
		options += " --model_to_brain"

	############# CREATE MODEL #############
	
	if args.create_model:
		### multiparallelize texts
		### todo: add here

		### preprocess ### train ### translate
		### todo: add locations
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

	############# FORMAT BRAIN DATA #############
	if args.format_data:
		format_cmd = "python format_for_subject.py --subject_number " + str(subj_num)
		os.system(format_cmd)

	############# MAKE SCRIPTS #############
	cmd = "python make_scripts.py"
	entire_cmd = cmd + " " + options + " " + get_residuals_and_make_scripts
	os.system(entire_cmd)

	############# MASTER BASH SCRIPTS #############
	# find script path
	if args.brain_to_model:
		direction = "brain2model_"
	else:
		direction = "model2brain_"

	if args.cross_validation:
		validate = "cv_"
	else:
		validate = "nocv_"

	model_type = str(direction) + str(validate) + "subj{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
	folder_name = model_type.format(
		args.subject_number, 
		args.language, 
		args.num_layers, 
		args.model_type, 
		args.which_layer, 
		args.agg_type
	)
	executable_path = "../decoding_scripts/" + str(folder_name) + "/" + str(folder_name) + ".sh"

	# make script executable
	cmd = "chmod +x " + str(executable_path)
	os.system(cmd)
	exe = "./" + str(executable_path) 
	os.system(exe)

	### wait for job dependency
	### todo: get individual jobs ids

	# for i in range(args.nbatches):
	# file = str(direction) + str(validate) + "subj{}_decoding_{}_of_{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
	# job_id = file.format(
	# 	args.subject_number, 
	# 	i, 
	# 	args.nbatches, 
	# 	args.language, 
	# 	args.num_layers, 
	# 	args.model_type, 
	# 	args.which_layer, 
	# 	args.agg_type
	# )
	# fname = '../decoding_scripts/' + str(folder_name) + '/' + str(job_id) + '.sh'

	############# CONCATENATE RMSE #############
	rmse = "python get_residuals.py" 
	entire_cmd = rmse + " " + options + " " + " " + get_residuals_and_make_scripts
	os.system(entire_cmd)

	############# PLOT VISUALIZATIONS #############
	plot = "python plot_residuals_location.py" 
	entire_cmd = plot + " " + options
	os.system(entire_cmd)

	############# PERMUTATION TESTING #############

	return

if __name__ == "__main__":
    main()