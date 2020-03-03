#!/bin/bash
#SBATCH -J ranking_word2vec 					# Job name
#SBATCH -p seas_dgx1 							# partition (queue)
#SBATCH --mem 2000 							# memory pool for all cores
#SBATCH -t 0-2:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../../outpt_ranking_word2vec.txt 		# File that STDOUT writes to
#SBATCH -e ../../../err_ranking_word2vec.txt		# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../batch_calculate_rank_across_sentences.py --cross_validation --model_to_brain --total_batches 100 --word2vec