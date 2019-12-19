import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm
from format_for_subject import get_modified_activations
from plot_residuals_location import clean_roi, clean_atlas

def shuffle_voxels_by_brain_region(activations, volmask, vals, labels, save_path=""):
	print("shuffling voxels by brain region...")
	num_sentences, num_voxels = activations.shape
	shuffled_activations = np.zeros((num_sentences, num_voxels))
	num_regions = len(set(labels))
	print("number of regions: " + str(num_regions))

	for sentence in tqdm(range(num_sentences)):
		sentence_activations = activations[sentence]
		for region in range(num_regions):
			activations_in_region_index = np.where(sentence_activations == region)[0]
			np.random.shuffle(activations_in_region_index)
			activations_in_region_values = sentence_activations[activations_in_region_index]
			np.put(shuffled_activations[sentence], activations_in_region_index, activations_in_region_values)
	
	print("saved file: " + str(save_path) + "activations.p")
	pickle.dump( shuffled_activations, open(save_path+"activations.p", "wb" ) )

	return shuffled_activations

def shuffle_voxels(activations, volmask, save_path=""):
	print("shuffling voxels...")
	num_sentences, num_voxels = activations.shape
	shuffled_activations = np.zeros((num_sentences, num_voxels))

	for sentence in tqdm(range(num_sentences)):
		sentence_activations = activations[0]

		np.random.shuffle(sentence_activations)
		shuffled_activations[sentence] = sentence_activations

	print("ACTIVATIONS SHAPE: " + str(activations.shape))
	print("SHUFFLED ACTIVATIONS SHAPE: " + str(shuffled_activations.shape))

	if activations.shape != shuffled_activations.shape:
		print("error in shuffling shape")
		exit()

	print("saved file: " + str(save_path) + "activations.p")
	pickle.dump( shuffled_activations, open(save_path+"activations.p", "wb" ) )
	return shuffled_activations

def main():
	# parse arguments
	argparser = argparse.ArgumentParser(description="random permutation of activations")
	argparser.add_argument("-subject_number", "--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	argparser.add_argument("-local",  "--local", action='store_true', default=False, help="True if running locally")
	argparser.add_argument("-permutation_region",  "--permutation_region", action='store_true', default=False, help="True if permutation by brain region")
	args = argparser.parse_args()

	print("getting arguments...")

	activations = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/activations.p", "rb" ) )
	volmask = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/volmask.p", "rb" ) )

	if args.permutation_region:
		atlas_vals = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/atlas_vals.p", "rb" ) )
		atlas_labels = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/atlas_labels.p", "rb" ) )

		atlas_labels_by_brain = clean_atlas(atlas_vals, atlas_labels)
		atlas_vals = np.reshape(atlas_vals, (len(atlas_vals),))

		save_location = f'../examplesGLM/subj{args.subject_number}/permutation_region_'
		shuffled_activations = shuffle_voxels_by_brain_region(activations, volmask, atlas_vals, atlas_labels_by_brain, save_path=save_location)
	else:
		save_location = f'../examplesGLM/subj{args.subject_number}/permutation_'
		shuffled_activations = shuffle_voxels(activations, volmask, save_path=save_location)

	modified_activations = get_modified_activations(shuffled_activations, volmask, save_path=save_location)

	print('done.')
	return

if __name__ == "__main__":
	main()