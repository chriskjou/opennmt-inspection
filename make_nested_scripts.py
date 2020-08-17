import os
import sys
import argparse
from tqdm import tqdm
import helper

def main():
	parser = argparse.ArgumentParser("make scripts for Odyssey cluster nested_cv")
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers", type=int, default=12)
	parser.add_argument("-glove", "--glove",  action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	parser.add_argument("-word2vec", "--word2vec",  action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	parser.add_argument("-bert", "--bert",  action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	parser.add_argument("-opennmt", "--opennmt",  action='store_true', default=False, help="True if initialize opennmt embeddings, False if not")
	parser.add_argument("-email", "--email", help="email associated with Odyssey cluster", type=str, default="ckjou@college.harvard.edu")
	parser.add_argument("-slurm_path", "--slurm_path", help="path for output/error files (default on home directory)", type=str, default="")
	parser.add_argument("-file_path", "--file_path", help="path to opennmt-inspection directory on the cluster", type=str, default="projects/")
	parser.add_argument("-embedding_path", "--embedding_path", help="path for embedding directory on the cluster", type=str, default="/n/shieber_lab/Lab/users/cjou/embeddings/")
	parser.add_argument("-rsa", "--rsa", action='store_true', default=False, help="True if RSA is used to generate residual values")
	args = parser.parse_args()

	print("creating directories...")
	if args.word2vec:
		model = "word2vec"
		num_layers = 1
	if args.glove:
		model = "glove"
		num_layers = 1
	if args.bert:
		model = "bert"
		num_layers = 12
	if not args.word2vec and not args.glove and not args.bert:
		model = "opennmt"
		num_layers = 4

	file_path = "../nested_cv"
	if args.rsa:
		file_path += "_rsa"

	if not os.path.exists(str(file_path) + "/"):
		os.makedirs(str(file_path) + "/")

	subjects = [1,2,4,5,7,8,9,10,11]
	for subj_num in subjects:
		folder_name = "nested_cv_{}_subj{}".format(model, subj_num)
		if not os.path.exists(str(file_path) + '/' + str(folder_name) + '/'):
			os.makedirs(str(file_path) + '/' + str(folder_name) + '/')

		print("generating scripts...")

		script_to_open = str(file_path) + '/' + folder_name + '/' + folder_name + '.sh'
		with open(script_to_open, "w") as rsh:
			rsh.write('''\
#!/bin/bash
for i in `seq 1 {}`; do
  sbatch "nested_cv_{}_subj{}_layer""$i"".sh" -H
done
'''.format(
		num_layers,
		model,
		subj_num
	)
)
		for num_layer in tqdm(range(1, num_layers + 1)):
			helper.create_nested_scripts(args, model, num_layer, subj_num)
	print("done.")

if __name__ == "__main__":
    main()