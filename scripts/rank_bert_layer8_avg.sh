#!/bin/bash
#SBATCH -J rank_bert_layer8_avg 				# Job name
#SBATCH -p serial_requeue  						# partition (queue)
#SBATCH --mem 10000 							# memory pool for all cores
#SBATCH -t 0-24:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../../err_rank_bert_layer8_avg.txt 		# File that STDOUT writes to
#SBATCH -e ../../../outpt_rank_bert_layer8_avg.txt		# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../batch_calculate_rank_across_sentences.py --which_layer 8 --cross_validation --bert --brain_to_model --total_batches 100