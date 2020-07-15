import os
import sys
import argparse
from tqdm import tqdm
import helper

def main():
	parser = argparse.ArgumentParser("make scripts for Odyssey cluster nested_cv")
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers", type=int, default=12)
	parser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	parser.add_argument("-glove", "--glove",  action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	parser.add_argument("-word2vec", "--word2vec",  action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	parser.add_argument("-bert", "--bert",  action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	parser.add_argument("-opennmt", "--opennmt",  action='store_true', default=False, help="True if initialize opennmt embeddings, False if not")
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

	folder_name = "nested_cv_{}_subj{}".format(model, args.subject_number)
	if not os.path.exists('../nested_cv/'):
		os.makedirs('../nested_cv/')
		
	if not os.path.exists('../nested_cv/' + str(folder_name) + '/'):
		os.makedirs('../nested_cv/' + str(folder_name) + '/')

	print("generating scripts...")

	script_to_open = '../nested_cv/' + folder_name + '/' + folder_name + '.sh'
	with open(script_to_open, "w") as rsh:
		rsh.write('''\
#!/bin/bash
for i in `seq 1 {}`; do
  sbatch "nested_cv_{}_subj{}_layer""$i"".sh" -H
done
'''.format(
		num_layers,
		model,
		args.subject_number
	)
)
	for num_layer in tqdm(range(1, num_layers + 1)):
		helper.create_nested_scripts(args, model, num_layer)
	print("done.")

if __name__ == "__main__":
    main()