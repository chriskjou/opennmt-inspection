import os
import sys
import argparse
from tqdm import tqdm

def save_script(args):
	if not os.path.exists('../decoding_scripts/'):
		os.makedirs('../decoding_scripts/')

	for i in range(args.nbatches):
		fname = '../decoding_scripts/subj{}_decoding_{}_of_{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}.sh'.format(args.subj_num, i, args.nbatches, args.language, args.num_layers, args.model_type, args.which_layer, args.agg_type)
		job_id = 'subj{}_decoding_{}_of_{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}'.format(args.subj_num, i, args.nbatches, args.language, args.num_layers, args.model_type, args.which_layer, args.agg_type)
		with open(fname, 'w') as rsh:
			rsh.write('''\
#!/bin/bash
#SBATCH -J {}  									# Job name
#SBATCH -p seas_dgx1 							# partition (queue)
#SBATCH --mem 10000 							# memory pool for all cores
#SBATCH -t 0-24:00 								# time (D-HH:MM)
#SBATCH --output=/n/home08/smenon # file output location
#SBATCH -o ../logs/outpt.txt 			# File that STDOUT writes to
#SBATCH -e ../logs/err.txt			# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skmenon@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate test

python ../opennmt-inspection/odyssey_decoding.py \
/n/scratchlfs/shieber_lab/users/smenon/embeddings/parallel/{}/{}layer-{}/{}/parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}.mat \
/n/scratchlfs/shieber_lab/users/fmri/subj{}/examplesGLM.mat \
subj{} \
{} \
{}
'''.format(job_id, args.language, args.num_layers, args.model_type, args.agg_type, args.language, args.num_layers, args.model_type, args.which_layer, args.agg_type, args.subj_num, args.subj_num, i, args.nbatches))

def main():
	# if len(sys.argv) != 3:
	# 	print("usage: python make_scripts.py -language -num_layers -brnn/rnn -which_layer -agg_type -subj_num -num_batches")
	# 	# example: python make_scripts.pe
	# 	exit()

	# usage: python make_scripts.py -language -num_layers -type -which_layer -agg_type -subj_num -num_batches"
	parser = argparse.ArgumentParser()

	#-db DATABSE -u USERNAME -p PASSWORD -size 20
	parser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	parser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	parser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	parser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	parser.add_argument("-subj_num", "--subj_num", help="fMRI subject number ([1:11])", type=int, default=1)
	parser.add_argument("-nbatches", "--nbatches", help="Total number of batches to run", type=int, default=100)

	args = parser.parse_args()

	languages = ['spanish'] #['spanish', 'german', 'italian', 'french', 'swedish']
	num_layers = [2] #[2, 4]
	model_type = ['brnn'] #['brnn', 'rnn']
	agg_type = ['avg', 'max', 'min', 'last']
	subj_num = [1]
	nbatches = 100

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
	if args.subj_num not in subj_num:
		print("invalid subj_num")
		exit()
	if args.which_layer not in list(range(args.num_layers)):
		print("invalid which_layer: which_layer must be between 1 and args.num_layers, inclusive")
		exit()

	print("generating scripts...")
	for lang in tqdm(languages):
		for nlayer in num_layers:
			for mtype in model_type:
				for atype in agg_type:
					for snum in subj_num:
						for layers in list(range(1, nlayer+1)):
							args.language = lang
							args.num_layers = nlayer
							args.model_type = mtype
							args.which_layer = layers
							args.agg_type = atype
							args.subj_num = snum
							args.nbatches = 100
							save_script(args)

	# residuals path (relative from opennmt):
	resid_path = '../residuals'
	if not os.path.isdir(resid_path):
		os.mdkir(resid_path)
	# embedding_layer = sys.argv[1]
	# subj_num = sys.argv[2]
	# num_batches = int(sys.argv[3])
	# save_script(args)
	# save_script(embedding_layer, subj_num, num_batches)
	print("done.")

if __name__ == "__main__":
    main()
