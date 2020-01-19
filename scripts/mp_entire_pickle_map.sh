#!/bin/bash
#SBATCH -J mp_entire_entire_pickle_map  		# Job name
#SBATCH -p serial_requeue 						# partition (queue)
#SBATCH -n 5 									# Number of cores requested
#SBATCH -N 1 									# Ensure that all cores are on one machine
#SBATCH --mem 10000 							# memory requested per node
#SBATCH -t 2-00:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../../outpt_mp_entire_entire_pickle_map.txt 		# File that STDOUT writes to
#SBATCH -e ../../../err_mp_entire_entire_pickle_map.txt		# File that STDERR writes to
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../mp_calculate_average_rank.py --embedding_layer /n/shieber_lab/Lab/users/cjou/embeddings/parallel/spanish/2layer-brnn/avg/parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg.mat  --model_to_brain -cross_validation --batch_num 0 --total_batches 100  