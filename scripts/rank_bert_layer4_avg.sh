#!/bin/bash
#SBATCH -J rank_bert_layer4_avg 				# Job name
#SBATCH -p seas_dgx1 							# partition (queue)
#SBATCH --mem 2000 								# memory pool for all cores
#SBATCH -t 0-24:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../../err_rank_bert_layer4_avg.txt 		# File that STDOUT writes to
#SBATCH -e ../../../outpt_rank_bert_layer4_avg.txt		# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../batch_calculate_rank_across_sentences.py --which_layer 4 --cross_validation --bert --model_to_brain --total_batches 100