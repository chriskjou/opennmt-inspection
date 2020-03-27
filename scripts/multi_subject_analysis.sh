#!/bin/bash
#SBATCH -J fdr_multi_subject_analysis  								# Job name
#SBATCH -p serial_requeue 											# partition (queue)
#SBATCH --mem 10000 												# memory pool for all cores
#SBATCH -t 0-24:00 													# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 							# file output location
#SBATCH -o ../../../outpt_fdr_multi_subject_analysis.txt 			# File that STDOUT writes to
#SBATCH -e ../../../err_fdr_multi_subject_analysis.txt			# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../significance_threshold.py  --embedding_layer /n/shieber_lab/Lab/users/cjou/embeddings/bert/layer1/avg.p --subjects 1,2,4,5,7,8,9,10,11 --bert -which_layer 1 --group_level
