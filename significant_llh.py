from tqdm import tqdm
import scipy.io
import pickle
import numpy as np
import sys
import argparse
import os
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# plt.switch_backend('agg')

def get_file(batch_num, total_batches):
	file_name = "all_batch{}of{}_bor_pxp.mat".format(batch_num, total_batches)
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
	print(np.array(bors.shape))
	print(pxps.shape)
	return np.array(bors), np.array(pxps)

def main():
	subjects = [1,2,4,5,7,8,9,10,11]
	num_layers = 18
	threshold = 0.01

	common_space = helper.load_common_space(subjects, local=True)
	voxel_coordinates = np.transpose(np.nonzero(common_space))
	# print(voxel_coordinates.shape)

	print("concatenating files...")
	bors, pxps = concatenate_files()
	print("BORS SHAPE: " + str(bors.shape))
	print("PXPS SHAPE: " + str(pxps.shape))
	print("BORS: " + str(bors))
	
	# plt.clf()
	# sns.set(style="darkgrid")
	# plt.figure(figsize=(16, 9))
	# _ = plt.hist(bors, bins='auto')
	# plt.ylabel("count")
	# plt.xlabel("BOR")
	# plt.savefig("../all_bors_hist.png", bbox_inches='tight')

	# print(np.sum(bors <= threshold))
	# print(np.sum(bors > threshold))
	# print(np.min(bors))

	# total = 0
	# all_pxps = []
	# for coord_index in tqdm(range(len(voxel_coordinates))):
	# 	x,y,z = voxel_coordinates[coord_index]
	# 	get_pxp = np.max(pxps[coord_index])
	# 	if get_pxp > .9:
	# 		total+=1
	# 	all_pxps.append(get_pxp)

	# plt.clf()
	# sns.set(style="darkgrid")
	# plt.figure(figsize=(16, 9))
	# _ = plt.hist(all_pxps, bins='auto')
	# # plt.xlim(0,1)
	# plt.ylabel("count")
	# plt.xlabel("maximum PXP per voxel")
	# plt.savefig("../all_pxp_values_hist.png", bbox_inches='tight')
	# print("TOTAL: " + str(total))
	# exit()

	a,b,c = common_space.shape
	mapped_space = np.zeros((a,b,c))
	all_layer_space = np.zeros((num_layers,a,b,c))
	print("ALL LAYER SPACE: " + str(all_layer_space.shape))

	# 121099
	print("creating maps...")
	for coord_index in tqdm(range(len(voxel_coordinates))):
		x,y,z = voxel_coordinates[coord_index]
		# if bors[coord_index] < threshold:
		# 	print(bors[coord_index])
		# 	print("HERE")
		# print(pxps[coord_index])
		# index = np.argmax(pxps[coord_index])
		print(pxps[coord_index])
		print(pxps[coord_index].shape)
		if np.max(pxps[index]) > 0.9:
			print(pxps[coord_index])
			print(pxps[coord_index].shape)
			mapped_space[x][y][z] = np.argmax(pxps[coord_index]) + 1
		# 
		# print(np.max(pxps[index]))
		for layer in range(num_layers):
			all_layer_space[layer][x][y][z] = pxps[coord_index][layer]

	print("MAPPED SPACE: " + str(mapped_space.shape))
	total_voxels = []
	for layer in range(1, num_layers+1):
		total = np.sum(mapped_space == layer)
		total_voxels.append(total)

	df = pd.DataFrame(
		{
			'layer': list(range(1, num_layers + 1 )),
			'num_voxels': total_voxels
		})

	# print(df.head())
	# sns.set(style="darkgrid")
	# plt.figure(figsize=(16, 9))
	# g = sns.catplot(x="layer", y="num_voxels", kind = "bar", color="cornflowerblue", data=df)
	# plt.savefig("../best_voxel_hist.png", bbox_inches='tight')
	# plt.show()
	
	scipy.io.savemat("../significant_pval_all_best_llh_by_voxel.mat", dict(metric = mapped_space))

	# for layer in range(num_layers):
	# 	scipy.io.savemat("../significant_all_best_llh_by_voxel_layer" + str(layer+1) + ".mat", dict(metric = all_layer_space[layer]))

	print("done.")


if __name__ == "__main__":
	main()