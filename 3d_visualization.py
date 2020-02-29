import numpy as np
import pickle
import os
import argparse
from nilearn import plotting, datasets, image
from tqdm import tqdm
import nibabel as nib
from nilearn import datasets
import helper

def plot_on_glass(data, file_name):
	# used from affine from demo (https://nilearn.github.io/auto_examples/01_plotting/plot_demo_glass_brain.html)
	# aff = np.array([[-3., 0., 0., 78.], [0.,3., 0.,-112.], [0.,0.,3.,-50.], [0.,0.,0.,1.]])
	# aff = np.array([[-2., 0., 0., 78.], [0.,2., 0.,-112.], [0.,0.,2.,-70.], [0.,0.,0.,1.]])
	aff = np.array([[-2., 0., 0., 0.], [0., 2., 0., 0.], [0., 0., 2., 0], [78., -112., -60., 1.]])
	# real_data = nib.affines.apply_affine(aff, data)
	# print("AFFINE TRANSFORM: " + str(real_data.shape))
	# print("NIFTI: " + str(new_image))
	new_image = nib.Nifti1Image(data, affine=aff)
	plotting.plot_glass_brain(new_image, output_file=file_name, colorbar=True, plot_abs=True, threshold='auto')
	plotting.show()
	return

def plot_interactive(data, file_name):
	print(np.shape(data))
	new_image = nib.Nifti1Image(data, affine=np.eye(4))
	view = plotting.view_img(new_image, threshold='auto')
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
	# argparser.add_argument('--rmse', type=str, help="Location of RMSE for entire brain (.p)", required=True)
	argparser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	argparser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	argparser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	argparser.add_argument("-subject_number", "--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	argparser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation", action='store_true', default=False)
	argparser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model", action='store_true', default=False)
	argparser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain", action='store_true', default=False)
	argparser.add_argument("-glove", "--glove", action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	argparser.add_argument("-word2vec", "--word2vec", action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("-bert", "--bert", action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("-rand_embed", "--rand_embed", action='store_true', default=False, help="True if initialize random embeddings, False if not")
	argparser.add_argument("-random",  "--random", action='store_true', default=False, help="True if add cross validation, False if not")
	argparser.add_argument("-permutation",  "--permutation", action='store_true', default=False, help="True if permutation, False if not")
	argparser.add_argument("-permutation_region", "--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	args = argparser.parse_args()

	print("getting arguments...")
	# rmses = args.rmse
	# file_name = rmses.split("/")[-1].split(".")[0]

	# get residuals
	# check conditions // can remove when making pipeline
	if args.brain_to_model and args.model_to_brain:
		print("select only one flag for brain_to_model or model_to_brain")
		exit()
	if not args.brain_to_model and not args.model_to_brain:
		print("select at least flag for brain_to_model or model_to_brain")
		exit()

	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

	# residual_file = sys.argv[1]
	if not args.word2vec and not args.glove and not args.bert and not args.random:
		specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
		file_name = specific_file.format(
			args.subject_number, 
			args.language, 
			args.num_layers, 
			args.model_type, 
			args.which_layer, 
			args.agg_type
		)
	else:
		file_name = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-{}_layer{}".format(args.subject_number, args.agg_type, args.which_layer)

	# file_loc = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "subj{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
	
	# file_name = file_loc.format(
	# 	args.subject_number, 
	# 	args.language, 
	# 	args.num_layers, 
	# 	args.model_type, 
	# 	args.which_layer, 
	# 	args.agg_type
	# )

	residual_file = "../rmses/concatenated-" + str(file_name) + ".p"

	# file_name = residual_file.split("/")[-1].split(".")[0]

	data = pickle.load( open( residual_file, "rb" ) )

	# get volmask
	subject_number = args.subject_number
	file_path = "../examplesGLM/subj{}/volmask.p".format(subject_number)
	volmask = pickle.load( open( file_path, "rb" ) )

	if not os.path.exists('../3d-brain/'):
		os.makedirs('../3d-brain/')

	print("transforming coordinates...")
	transform_data = helper.transform_coordinates(data, volmask, save_path="../3d-brain/"+file_name, metric="rmse")
	print("ORIGINAL DATA: " + str(len(data)))
	print("TRANSFORMED DATA: " + str(transform_data.shape))

	print("plotting data...")
	f_name = "../3d-brain/" + file_name + "-glass-brain.png"
	plot_on_glass(transform_data, f_name)
	# plot_interactive(transform_data, file_name)
	# plot_roi(transform_data, file_name)
	print('done.')
	return

if __name__ == "__main__":
    main()