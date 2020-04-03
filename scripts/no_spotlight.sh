#!/bin/bash
#SBATCH -J no_spotlight  								# Job name
#SBATCH -p serial_requeue 						# partition (queue)
#SBATCH --mem 5000 								# memory pool for all cores
#SBATCH -t 0-5:00									# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../../outpt_no_spotlight.txt 			# File that STDOUT writes to
#SBATCH -e ../../../err_no_spotlight.txt				# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../no_spotlight_decoding.py --embedding_layer /n/shieber_lab/Lab/users/cjou/embeddings/bert/layer1/avg.p --subject_mat_file /n/shieber_lab/Lab/users/cjou/fmri/subj1/examplesGLM.mat  --model_to_brain   --cross_validation  --subject_number 1 --which_layer 1     --bert    
