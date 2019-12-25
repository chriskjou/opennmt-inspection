#!/bin/bash
#SBATCH -J index_single_batch_rank_test		  		# Job name
#SBATCH -p seas_dgx1 							# partition (queue)
#SBATCH --mem 10000 							# memory pool for all cores
#SBATCH -t 0-24:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../rank_logs/outpt_index_single_batch_rank_test.txt 			# File that STDOUT writes to
#SBATCH -e ../../rank_logs/err_index_single_batch_rank_test.txt			# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python calculate_average_rank.py --embedding_layer /n/shieber_lab/Lab/users/cjou/embeddings/parallel/spanish/2layer-brnn/avg/parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg.mat --model_to_brain --total_batches 100