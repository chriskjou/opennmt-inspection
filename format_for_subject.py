# generate subject x's .p files from MAT files in ../embeddings folder

import scipy.io
from tqdm import tqdm
import pickle
import numpy as np
import sys
import argparse

# get initial information from MATLAB
def get_activations(info, save_path=""):
	print("getting activations...")
	mat = scipy.io.loadmat(info)

	activations = mat["examples_sentences"]
	volmask = mat["volmask"]
	atlas_vals = mat["multimask_aal"]
	roi_vals = mat["multimask_group"]
	roi_labels = mat["labels_langloc"]
	atlas_labels = mat["labels_aal"]

	print("writing to file...")
	pickle.dump( activations, open(save_path+"activations.p", "wb" ) )
	pickle.dump( volmask, open(save_path+"volmask.p", "wb" ) )

	pickle.dump(atlas_vals, open(save_path+"atlas_vals.p", "wb") )
	pickle.dump(atlas_labels, open(save_path+"atlas_labels.p", "wb") )

	pickle.dump(roi_vals, open(save_path+"roi_vals.p", "wb") )
	pickle.dump(roi_labels, open(save_path+"roi_labels.p", "wb") )

	print("finished.")
	return activations, volmask


def get_modified_activations(activations, volmask, save_path=""):
	i,j,k = volmask.shape
	nonzero_pts = np.transpose(np.nonzero(volmask))
	modified_activations = []
	for sentence_activation in tqdm(activations):
		one_sentence_act = np.zeros((i,j,k))
		for pt in range(len(nonzero_pts)):
			x,y,z = nonzero_pts[pt]
			one_sentence_act[int(x)][int(y)][int(z)] = sentence_activation[pt]
		modified_activations.append(one_sentence_act)
	pickle.dump( modified_activations, open(save_path + "modified_activations.p", "wb" ) )
	return modified_activations


def main():
    file_description = ("This is used to convert the .MAT embeddings " +
                        "from the subjects to a training-readable format. \n" +
                        "Please use this file before calling any decoding functions.")
    argparser = argparse.ArgumentParser(description=file_description)
    subj_num_description = "The subject(s) # in exampleGLM whose .mat files need to be processed"
    argparser.add_argument("--subject_number", type=int, default=[1], nargs='+', help=subj_num_description)
    args = argparser.parse_args()
    print(args.subject_number)
    for subj_num in args.subject_number:
        save_location = f'../examplesGLM/subj{subj_num}/'
        glm_sentences_path = save_location + 'examplesGLM.mat'
        activations, volmask = get_activations(glm_sentences_path, save_path=save_location)
        print("saved activations.")
        modified_activations = get_modified_activations(activations, volmask, save_path=save_location)
        print("saved modified activations.")
    print("done.")


if __name__ == "__main__": main()
