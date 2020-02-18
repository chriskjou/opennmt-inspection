#!/bin/bash
#SBATCH -J rsa_test 							# Job name
#SBATCH -p serial_requeue 						# partition (queue)
#SBATCH --mem 5000 								# memory pool for all cores
#SBATCH -t 0-5:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../../outpt_rsa_test.txt 			# File that STDOUT writes to
#SBATCH -e ../../../err_rsa_test.txt				# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../rsa.py --embedding_layer /n/shieber_lab/Lab/users/cjou/embeddings/parallel/spanish/2layer-brnn/avg/parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg.mat --subject_mat_file /n/shieber_lab/Lab/users/cjou/fmri/subj1/examplesGLM.mat  --brain_to_model --subject_number 1 --batch_num 0 --total_batches 100 --rsa
