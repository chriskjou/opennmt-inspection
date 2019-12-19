import os
import sys
import argparse
from tqdm import tqdm

def save_script(args):
	if args.local:
		if not os.path.exists('../decoding_scripts/'):
			os.makedirs('../decoding_scripts/')
	else:
		if not os.path.exists('../../decoding_scripts/'):
			os.makedirs('../../decoding_scripts/')

	# file name assignments
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
		plabel = "permutation"
	else:
		plabel = ""

	if args.permutation_region:
		prlabel = "permutation_region"
	else:
		prlabel = ""

	# create subfolder
	model_type = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "subj{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
	folder_name = model_type.format(
		args.subject_number, 
		args.language, 
		args.num_layers, 
		args.model_type, 
		args.which_layer, 
		args.agg_type
	)
	print(folder_name)

	if args.local:
		if not os.path.exists('../decoding_scripts/' + str(folder_name) + '/'):
			os.makedirs('../decoding_scripts/' + str(folder_name) + '/')
		script_to_open = "../decoding_scripts/" + str(folder_name) + "/" + str(folder_name) + ".sh"
	else:
		if not os.path.exists('../../decoding_scripts/' + str(folder_name) + '/'):
			os.makedirs('../../decoding_scripts/' + str(folder_name) + '/')
		script_to_open = "../../decoding_scripts/" + str(folder_name) + "/" + str(folder_name) + ".sh"

	# make master script
	with open(script_to_open, "w") as rsh:
		rsh.write('''\
#!/bin/bash
for i in `seq 0 99`; do
  sbatch "{}{}{}{}{}{}{}{}{}subj{}_decoding_""$i""_of_{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}.sh" -H
done
'''.format(
		plabel,
		prlabel,
		rlabel,
		elabel,
		glabel,
		w2vlabel,
		bertlabel,
		direction,
		validate,
		args.subject_number, 
		args.total_batches, 
		args.language, 
		args.num_layers, 
		args.model_type, 
		args.which_layer, 
		args.agg_type
	)
)

	# break into batches
	for i in range(args.total_batches):
		file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "subj{}_decoding_{}_of_{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
		job_id = file.format(
			args.subject_number, 
			i, 
			args.total_batches, 
			args.language, 
			args.num_layers, 
			args.model_type, 
			args.which_layer, 
			args.agg_type
		)

		if args.local:
			fname = '../decoding_scripts/' + str(folder_name) + '/' + str(job_id) + '.sh'
		else:
			fname = '../../decoding_scripts/' + str(folder_name) + '/' + str(job_id) + '.sh'

		with open(fname, 'w') as rsh:
			pflag = "" if (plabel == "") else "--" + str(plabel)
			prflag = "" if (prlabel == "") else "--" + str(prlabel)
			rflag = "" if (rlabel == "") else "--" + str(rlabel)
			gflag = "" if (glabel == "") else "--" + str(glabel)
			w2vflag = "" if (w2vlabel == "") else "--" + str(w2vlabel)
			bertflag = "" if (bertlabel == "") else "--" + str(bertlabel)
			eflag = "" if (elabel == "") else "--" + str(elabel)
			rsh.write('''\
#!/bin/bash
#SBATCH -J {0}  								# Job name
#SBATCH -p seas_dgx1 							# partition (queue)
#SBATCH --mem 10000 							# memory pool for all cores
#SBATCH -t 0-24:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../logs/outpt_{0}.txt 			# File that STDOUT writes to
#SBATCH -e ../../logs/err_{0}.txt				# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../../projects/opennmt-inspection/odyssey_decoding.py \
--embedding_layer /n/scratchlfs/shieber_lab/users/cjou/embeddings/parallel/{1}/{2}layer-{3}/{4}/parallel-english-to-{1}-model-{2}layer-{3}-pred-layer{5}-{4}.mat \
--subject_mat_file /n/scratchlfs/shieber_lab/users/fmri/subj{6}/examplesGLM.mat \
--brain_to_model {7} \
--cross_validation {8} \
--subject_number {6} \
--batch_num {9} \
--total_batches {10} \
{11} {12} {13} {14} {15} {16} {17}
'''.format(
		job_id, 
		args.language, 
		args.num_layers, 
		args.model_type, 
		args.agg_type, 
		args.which_layer, 
		args.subject_number, 
		args.brain_to_model,
		args.cross_validation,
		i, 
		args.total_batches,
		rflag,
		eflag,
		gflag,
		w2vflag,
		bertflag,
		pflag,
		prflag
	)
)

def main():
	# if len(sys.argv) != 3:
	# 	print("usage: python make_scripts.py -language -num_layers -brnn/rnn -which_layer -agg_type -subject_number -num_batches")
	# 	# example: python make_scripts.pe
	# 	exit()

	# usage: python make_scripts.py -language -num_layers -type -which_layer -agg_type -subject_number -num_batches"
	parser = argparse.ArgumentParser("make scripts for Odyssey cluster")
	parser.add_argument("-total_batches", "--total_batches", help="Total number of batches to run", type=int, default=100)
	parser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	parser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	parser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	parser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	parser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	parser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation", action='store_true', default=False)
	parser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model", action='store_true', default=False)
	parser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain", action='store_true', default=False)
	parser.add_argument("-random",  "--random", action='store_true', default=False, help="True if initialize random brain activations, False if not")
	parser.add_argument("-rand_embed",  "--rand_embed", action='store_true', default=False, help="True if initialize random embeddings, False if not")
	parser.add_argument("-glove", "--glove", action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	parser.add_argument("-word2vec", "--word2vec", action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	parser.add_argument("-bert", "--bert", action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	parser.add_argument("-permutation", "--permutation", action='store_true', default=False, help="True if permutation, False if not")
	parser.add_argument("--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	parser.add_argument("-local", "--local", action='store_true', default=False, help="True if running locally, False if not")
	args = parser.parse_args()

	languages = ['spanish', 'german', 'italian', 'french', 'swedish']
	num_layers = [2, 4]
	model_type = ['brnn', 'rnn']
	agg_type = ['avg', 'max', 'min', 'last']
	subject_number = list(range(1,12))
	# nbatches = 100
	
	# check conditions
	if args.brain_to_model and args.model_to_brain:
		print("select only one flag for brain_to_model or model_to_brain")
		exit()
	if not args.brain_to_model and not args.model_to_brain:
		print("select at least flag for brain_to_model or model_to_brain")
		exit()
	if args.word2vec and args.glove and args.bert and args.rand_embed:
		print("select at most one flag for glove, word2vec, bert, and random")
		exit()

	# check
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
	if args.subject_number not in subject_number:
		print("invalid subject_number")
		exit()
	if args.which_layer not in list(range(1, args.num_layers+1)):
		print("invalid which_layer: which_layer must be between 1 and args.num_layers, inclusive")
		exit()

	print("generating scripts...")
	# for lang in tqdm(languages):
	# 	for nlayer in num_layers:
	# 		for mtype in model_type:
	# 			for atype in agg_type:
	# 				for snum in subject_number:
	# 					for layers in list(range(1, nlayer+1)):
	# 						args.language = lang
	# 						args.num_layers = nlayer
	# 						args.model_type = mtype
	# 						args.which_layer = layers
	# 						args.agg_type = atype
	# 						args.subject_number = snum
	# 						args.nbatches = 100
	# 						args.cross_validation = True
	# 						args.brain_to_model = True
	save_script(args)

	# residuals path (relative from opennmt):
	resid_path = '../residuals'
	if not os.path.isdir(resid_path):
		os.mdkir(resid_path)
	# embedding_layer = sys.argv[1]
	# subject_number = sys.argv[2]
	# num_batches = int(sys.argv[3])
	# save_script(args)
	# save_script(embedding_layer, subject_number, num_batches)
	print("done.")

if __name__ == "__main__":
    main()
