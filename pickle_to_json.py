import json
import pickle
import numpy as np
import time
import gc
from tqdm import tqdm

def convert_pickle_to_json(filename):
	with open("../true_spotlights/" + filename + ".p", 'rb') as f:
		data = pickle.load(f)

	with open(filename + ".json", 'w') as f:
		json.dump(np.array(data).tolist(), f)

	return

def load_pickle(filename):
	with open("../true_spotlights/" + filename + ".p", 'rb') as f:
		data = pickle.load(f)
	return data

def load_json(filename):
	with open(filename + ".json", 'r') as f:
		data = json.load(f)
	return data

def convert_pickle_to_binary(filename, filepath):
	with open("/n/shieber_lab/Lab/users/cjou/predictions/" + filename + ".p", 'rb') as f:
		data = pickle.load(f)

	f = open(filepath + filename + ".p", "wb")
	pickle.dump(data, f, protocol=-1)
	# pickle.HIGHEST_PROTOCOL
	f.close()

	# f = open(filename + "-binary-cp.p", "wb") 
	# cPickle.dump(data, f, protocol=-1)
	# f.close()
	return

def load_binary_pickle(filename):
	f = open(filename + "-binary.p", "rb")
	pickle.load(f)
	f.close()
	return

def load_cpickle(filename):
	f = open(filename + "-binary-cp.p", "rb")
	cPickle.load(f)
	f.close()
	return

def main():
	# gc.disable()

	# filename = "model2brain_nocv_-subj1-parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg_residuals_part{}of100-true-spotlights"

	# print("true spotlights")
	# for i in tqdm(range(100)):
	# 	with_file_number = filename.format(i)
	# 	convert_pickle_to_binary(with_file_number, "/n/shieber_lab/Lab/users/cjou/true_spotlights_bin/")

	filename = "model2brain_nocv_-subj1-parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg_residuals_part{}of100-decoding-predictions"

	print("decoding predictions")
	for i in tqdm(range(100)):
		with_file_number = filename.format(i)
		convert_pickle_to_binary(with_file_number, "/n/shieber_lab/Lab/users/cjou/predictions_bin/")
	# print("convert file...")
	# convert_pickle_to_json(filename)


	# convert_pickle_to_cpickle(filename)

	# start = time.time()
	# data = load_binary_pickle(filename)
	# end = time.time()
	# print("loading binary pickle: " + str(end-start))

	# start = time.time()
	# data = load_pickle(filename)
	# end = time.time()
	# print("loading pickle: " + str(end-start))

	# start = time.time()
	# data = load_cpickle(filename)
	# end = time.time()
	# print("loading cpickle: " + str(end-start))

	# gc.enable()
	print("done.")

if __name__ == "__main__":
	main()