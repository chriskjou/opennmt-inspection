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

def reduce_type(filename, filetype, filepath):
	if filetype == "predictions":
		get_path = "/n/shieber_lab/Lab/users/cjou/predictions/"
	if filetype == "true_spotlights":
		get_path = "/n/shieber_lab/Lab/users/cjou/true_spotlights/"

	with open( get_path + filename + ".p", 'rb') as f:
		data = pickle.load(f)

	f = open(filepath + filename + ".p", "wb")
	if filetype == "true_spotlights":
		modified_data = [sub.astype(np.float32) for elem in data for sub in elem]
	if filetype == "predictions":
		modified_data = [elem.astype(np.float32) for elem in data]
	pickle.dump(modified_data, f)
	# pickle.HIGHEST_PROTOCOL
	f.close()
	return

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
	### DOWNSIZE
	filename = "model2brain_nocv_-subj1-parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg_residuals_part{}of100-true-spotlights"

	print("true spotlights")
	for i in tqdm(range(100)):
		with_file_number = filename.format(i)
		reduce_type(with_file_number, "true_spotlights", "/n/shieber_lab/Lab/users/cjou/true_spotlights_32/")

	filename = "model2brain_nocv_-subj1-parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg_residuals_part{}of100-decoding-predictions"

	print("decoding predictions")
	# for i in tqdm(range(100)):
	# 	with_file_number = filename.format(i)
	# 	reduce_type(with_file_number, "predictions", "/n/shieber_lab/Lab/users/cjou/predictions_32/")
	### DOWNSIZE

	print("done.")

if __name__ == "__main__":
	main()