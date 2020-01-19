import pickle
import numpy as np
import time
import gc
from tqdm import tqdm
from multiprocessing import Pool

def convert_pickle_to_memmap(filename, filepath):
	with open(file_path + filename + ".p", 'rb') as f:
		data = pickle.load(f)

	return

def get_data(filename):
	global VOXEL_NUMBER
	fp = np.memmap(filename, dtype='float32', mode='r')
	VOXEL_NUMBER, num_sentences, act = int(fp[0]), int(fp[1]), int(fp[2])
	padding = num_sentences * act
	fp = fp[padding:].reshape((VOXEL_NUMBER, num_sentences, act))
	return fp

def open_memmap(i):
	file_name = "/n/shieber_lab/Lab/users/cjou/predictions_memmap/brain2model_cv_-subj1-parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg_residuals_part0of100.dat"
	memmap = get_data(file_name)
	# return memmap

def open_pickle(i):
	file = "/n/shieber_lab/Lab/users/cjou/predictions_od32/brain2model_nocv_-subj1-parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg_residuals_part0of100-decoding-predictions.p"
	pickle_contents = pickle.load(open(file, "rb"))
	# return pickle_contents

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
	pool = Pool()
	pool.map(open_memmap,[1,2,3,4,5,6,7,8])
	end = time.time()
	print("memmap: " + str(end - start))

	start = time.time()
	pool = Pool()
	pool.map(open_pickle,[1,2,3,4,5,6,7,8])
	end = time.time()
	print("pickle: " + str(end - start))

	print("done.")

if __name__ == "__main__":
	main() 