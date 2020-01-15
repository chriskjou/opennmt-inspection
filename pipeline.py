import sys
import subprocess
import os
from tqdm import tqdm
import helper 

# global variables
languages = ['spanish', 'german', 'italian', 'french', 'swedish']
num_layers = [2, 4]
model_type = ['brnn', 'rnn']
agg_type = ['avg', 'max', 'min', 'last']
subj_num = range(1,12)
nbatches = 100

def main():

	############# GET ARGUMENTS BELOW #############

	parser = argparse.ArgumentParser(description="entire OpenNMT pipeline: data prep, model, decoding, visualization")
	
	# model type
	parser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	parser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	parser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	parser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	parser.add_argument("-nbatches", "--nbatches", help="Total number of batches to run", type=int, default=100)
	
	# subject for brain data
	parser.add_argument("-subj_num", "--subj_num", help="fMRI subject number ([1:11])", type=int, default=1)
	parser.add_argument("-format_data", "--format_data", help="Format fMRI data", action='store_true', default=False)
	
	# opennmt model
	parser.add_argument("-create_model", "--create_model", help="create OpenNMT prediction model", action='store_true', default=False)
	
	# initializations
	parser.add_argument("-random", "--random",  action='store_true', default=False, help="True if add cross validation, False if not")
	parser.add_argument("-rand_embed",  "--rand_embed", action='store_true', default=False, help="True if initialize random embeddings, False if not")
	parser.add_argument("-glove", "--glove", action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	parser.add_argument("-word2vec", "--word2vec", action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	parser.add_argument("-bert", "--bert", action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	parser.add_argument("-permutation", "--permutation", action='store_true', default=False, help="True if permutation, False if not")
	parser.add_argument("-permutation_region", "--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	
	# evaluation metrics
	parser.add_argument("-decoding", "--decoding", action='store_true', default=False, help="True if decoding, False if not")
	parser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation", action='store_true', default=False)
	parser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model", action='store_true', default=False)
	parser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain", action='store_true', default=False)
	parser.add_argument("-fdr", "--fdr", action='store_true', default=False, help="True if FDR, False if not")
	parser.add_argument("-rank", "--rank", action='store_true', default=False, help="True if rank, False if not")
	parser.add_argument("-llh", "--llh", action='store_true', default=False, help="True if likelihood, False if not")

	parser.add_argument("-local", "--local", action='store_true', default=False, help="True if running locally, False if not")

	args = parser.parse_args()

	############# VALIDATE ARGUMENTS #############

	helper.validate_arguments()

	############# CREATE LABELS #############

	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

	############# GLOBAL OPTIONS #############

	get_residuals_and_make_scripts, options = helper.generate_options(args)

	############# CREATE OPENNMT MODEL #############
	
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

	############# MAKE SCRIPTS  #############

	if args.decoding or args.rank or args.llh:
		cmd = "python make_scripts.py"
		entire_cmd = cmd + " " + options + " " + get_residuals_and_make_scripts
		os.system(entire_cmd)

		model_type = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "subj{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
		# model_type = str(rlabel) + str(direction) + str(validate) + "subj{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
		folder_name = model_type.format(
			args.subject_number, 
			args.language, 
			args.num_layers, 
			args.model_type, 
			args.which_layer, 
			args.agg_type
		)
		executable_path = "../../decoding_scripts/" + str(folder_name) + "/" + str(folder_name) + ".sh"

		# make script executable
		cmd = "chmod +x " + str(executable_path)
		os.system(cmd)
		exe = "sbatch " + str(executable_path) 
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

		############# EVALUATION #############

		# 1. RMSE

		# CONCATENATE RMSE 
		rmse = "python get_residuals.py" 
		entire_cmd = rmse + " " + options + " " + " " + get_residuals_and_make_scripts
		os.system(entire_cmd)

		# CONVERT CONCATENATE TO MATLAB 
		convert = "python convert_np_to_matlab.py"
		entire_cmd = convert + " " + options + " " + " " + get_residuals_and_make_scripts + " -local"
		os.system(entire_cmd)

		# 2. AVERAGE RANK

		# RUN AVERAGE RANK

		# CONCATENATE AVERAGE RANK

		# CONVERT AVERAGE RANK TO MATLAB

	if args.fdr:
		cmd = "python significance_threshold.py"
		entire_cmd = cmd + " " + options + " " + get_residuals_and_make_scripts
		os.system(entire_cmd)

	############# VISUALIZATIONS #############
	# change to 3d

	# save to MATLAB

	# plot = "python plot_residuals_location.py" 
	# entire_cmd = plot + " " + options
	# os.system(entire_cmd)

	return

if __name__ == "__main__":
    main()