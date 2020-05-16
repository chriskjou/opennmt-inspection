import pickle
import argparse
import pandas as pd 
from tqdm import tqdm
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def main():
	parser = argparse.ArgumentParser("plot bert neurosynth rsa")
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers", type=int, default=12)
	parser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	args = parser.parse_args()

	corrs = []
	print("getting files...")
	for layer in tqdm(range(1, args.num_layers+1)):
		file_name = "bert_avg_layer{}_subj{}".format(layer, args.subject_number)
		loc = "../rsa_neurosynth/" + file_name + ".p"

		file_contents = pickle.load(open(loc, "rb"))
		corrs.append(file_contents)

	corrs = np.array(corrs)
	print(corrs.shape)
	print("done.")

if __name__ == "__main__":
    main()