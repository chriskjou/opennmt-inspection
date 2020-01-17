import pickle
import numpy as np
import time
import gc
from tqdm import tqdm

def convert_pickle_to_memmap(filename, filepath):
	with open(file_path + filename + ".p", 'rb') as f:
		data = pickle.load(f)

	return

def main():
	# create temp memmap


	# add files into memmap
	filename = "model2brain_nocv_-subj1-parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg_residuals_part{}of100-true-spotlights"

	print("true spotlights")
	for i in tqdm(range(100)):
		with_file_number = filename.format(i)
		convert_pickle_to_binary(with_file_number, "/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/")

	filename = "model2brain_nocv_-subj1-parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg_residuals_part{}of100-decoding-predictions"

	print("decoding predictions")
	for i in tqdm(range(100)):
		with_file_number = filename.format(i)
		convert_pickle_to_binary(with_file_number, "/n/shieber_lab/Lab/users/cjou/predictions_od32/")

	del fp
	print("done.")

if __name__ == "__main__":
	main() 