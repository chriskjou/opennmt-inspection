#!/bin/bash
#SBATCH -J ranking_permutation_bert 			# Job name
#SBATCH -p serial_requeue  						# partition (queue)
#SBATCH --mem 10000 							# memory pool for all cores
#SBATCH -t 0-10:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../../outpt_ranking_permutation_bert.txt 	# File that STDOUT writes to
#SBATCH -e ../../../err_ranking_permutation_bert.txt		# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../batch_calculate_rank_across_sentences.py --cross_validation --model_to_brain --total_batches 100 --bert --permutation_region