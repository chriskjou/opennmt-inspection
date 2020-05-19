import os
import sys
import argparse
from tqdm import tqdm
import helper

def main():
	parser = argparse.ArgumentParser("make scripts for Odyssey cluster bert neurosynth rsa")
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers", type=int, default=12)
	parser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	parser.add_argument("--null",  action='store_true', default=True, help="True if calculate significance, False if not")
	args = parser.parse_args()

	print("creating directories...")
	folder_name = "neurosynth_rsa_bert_subj{}".format(args.subject_number)
	if not os.path.exists('../neurosynth_rsa/'):
		os.makedirs('../neurosynth_rsa/')
		
	if not os.path.exists('../neurosynth_rsa/' + str(folder_name) + '/'):
		os.makedirs('../neurosynth_rsa/' + str(folder_name) + '/')

	print("generating scripts...")

	script_to_open = '../neurosynth_rsa/' + folder_name + '/' + folder_name + '.sh'
	with open(script_to_open, "w") as rsh:
		rsh.write('''\
#!/bin/bash
for i in `seq 1 12`; do
  sbatch "neurosynth_rsa_bert_subj{}_layer""$i"".sh" -H
done
'''.format(
		args.subject_number
	)
)
	for num_layer in tqdm(range(1, args.num_layers + 1)):
		helper.create_neurosynth_rsa_script(args, num_layer)
	print("done.")

if __name__ == "__main__":
    main()