from tqdm import tqdm
import scipy.io
import pickle
import numpy as np
import sys
import argparse
import os
import helper

def get_file(batch_num, total_batches):
	file_name = "batch{}of{}_bor_pxp.mat".format(batch_num, total_batches)
	file_contents = scipy.io.loadmat("../llh_mat/" + str(file_name))["llh"]
	bor = file_contents["bor"][0][0][0]
	pxp = file_contents["pxp"][0][0]
	return bor, pxp

def concatenate_files(total_batches=100):
	bors = []
	pxps = []
	for i in tqdm(range(total_batches)):
		bor, pxp = get_file(i, total_batches)
		bors.extend(bor)
		pxps.extend(pxp)
	return np.array(bors), np.array(pxps)

def main():
	subjects = [1,2,4,5,7,8,9,10,11]
	num_layers = 12
	threshold = 0.01

	common_space = helper.load_common_space(subjects, local=True)
	voxel_coordinates = np.transpose(np.nonzero(common_space))
	print(voxel_coordinates.shape)

	print("concatenating files...")
	bors, pxps = concatenate_files()
	print("BORS SHAPE: " + str(bors.shape))
	print("PXPS SHAPE: " + str(pxps.shape))

	a,b,c = common_space.shape
	mapped_space = np.zeros((a,b,c))
	all_layer_space = np.zeros((num_layers,a,b,c))
	print("ALL LAYER SPACE: " + str(all_layer_space.shape))

	# 121099
	print("creating maps...")
	for coord_index in tqdm(range(len(voxel_coordinates))):
		x,y,z = voxel_coordinates[coord_index]
		if bors[coord_index] >= threshold:
			mapped_space[x][y][z] = np.argmax(pxps[coord_index]) + 1
			for layer in range(num_layers):
				all_layer_space[layer][x][y][z] = pxps[coord_index][layer]

	print("MAPPED SPACE: " + str(mapped_space.shape))
	scipy.io.savemat("../significant_bert_best_llh_by_voxel.mat", dict(metric = mapped_space))

	for layer in range(num_layers):
		scipy.io.savemat("../significant_bert_best_llh_by_voxel_layer" + str(layer+1) + ".mat", dict(metric = all_layer_space[layer]))

	print("done.")


if __name__ == "__main__":
	main()