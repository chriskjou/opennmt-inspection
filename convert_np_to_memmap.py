import pickle
import numpy as np
import time
import gc
from tqdm import tqdm

def convert_pickle_to_memmap(filename, filepath):
	with open(file_path + filename + ".p", 'rb') as f:
		data = pickle.load(f)

	return

def open_memmap():
	file_name = "/n/shieber_lab/Lab/users/cjou/predictions_od32/memmap-version.dat"
	memmap = np.memmap(file_name, dtype='float32', mode='r', shape=(1513, 240, 500))
	return memmap

def open_pickle():
	file = "/n/shieber_lab/Lab/users/cjou/predictions_od32/model2brain_cv_-subj1-parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg_residuals_part9of100-decoding-predictions.p"
	pickle_contents = pickle.load(open(file, "rb"))
	return pickle_contents

def main():
	# create temp memmap


	# # add files into memmap
	# filename = "model2brain_nocv_-subj1-parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg_residuals_part{}of100-true-spotlights"

	# print("true spotlights")
	# for i in tqdm(range(100)):
	# 	with_file_number = filename.format(i)
	# 	convert_pickle_to_binary(with_file_number, "/n/shieber_lab/Lab/users/cjou/true_spotlights_od32/")

	# filename = "model2brain_nocv_-subj1-parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg_residuals_part{}of100-decoding-predictions"

	# print("decoding predictions")
	# for i in tqdm(range(100)):
	# 	with_file_number = filename.format(i)
	# 	convert_pickle_to_binary(with_file_number, "/n/shieber_lab/Lab/users/cjou/predictions_od32/")

	# del fp

	start = time.time()
	contents = open_memmap()
	end = time.time()
	print("memmap: " + str(end - start))

	start = time.time()
	contents = open_pickle ()
	end = time.time()
	print("pickle: " + str(end - start))

	print("done.")

if __name__ == "__main__":
	main() 