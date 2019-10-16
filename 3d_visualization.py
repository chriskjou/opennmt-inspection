import torch
import numpy as np
import pickle
import os
import argparse
from nilearn import plotting, datasets
from tqdm import tqdm
import nibabel as nib
from nilearn import datasets

def transform_coordinates(rmses, volmask, save_path=""):
	i,j,k = volmask.shape
	nonzero_pts = np.transpose(np.nonzero(volmask))
	modified_rmses = np.zeros((i,j,k))
	for pt in tqdm(range(len(nonzero_pts))):
		x,y,z = nonzero_pts[pt]
		modified_rmses[int(x)][int(y)][int(z)] = rmses[pt]
	pickle.dump( modified_rmses, open(save_path + "-transform-rmse.p", "wb" ) )
	return modified_rmses

def plot_on_glass(data, file_name):
	print(np.shape(data))
	new_image = nib.Nifti1Image(data, affine=np.eye(4))
	plotting.plot_glass_brain(new_image, threshold=0)
	plotting.show()
	return

def plot_interactive(data, file_name):
	new_image = nib.Nifti1Image(data, affine=np.eye(4))
	view = plotting.view_img(new_image, threshold=0)
	view.open_in_browser()
	return

def plot_roi(data, file_name):
	dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
	atlas_filename = dataset.maps
	print('Atlas ROIs are located at: %s' % atlas_filename)
	plotting.plot_roi(atlas_filename, title="harvard oxford altas")
	plotting.show()
	return

def main():
	# parse arguments
	argparser = argparse.ArgumentParser(description="plot RMSE on 3d brain")
	argparser.add_argument('--rmse', type=str, help="Location of RMSE for entire brain (.p)", required=True)
	argparser.add_argument('--subject_number', type=int, help="Subject number", required=True)
	args = argparser.parse_args()

	print("getting arguments...")
	rmses = args.rmse
	file_name = rmses.split("/")[-1].split(".")[0]
	data = pickle.load( open( rmses, "rb" ) )

	# get volmask
	subject_number = args.subject_number
	file_path = "../examplesGLM/subj{}/volmask.p".format(subject_number)
	volmask = pickle.load( open( file_path, "rb" ) )

	if not os.path.exists('../3d-brain/'):
		os.makedirs('../3d-brain/')

	print("transforming coordinates...")
	transform_data = transform_coordinates(data, volmask, "../3d-brain/")

	print("plotting data...")
	# plot_on_glass(transform_data, file_name)
	plot_interactive(transform_data, file_name)
	# plot_roi(transform_data, file_name)
	print('done.')
	return

if __name__ == "__main__":
    main()