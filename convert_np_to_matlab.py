import numpy as np
import argparse
from tqdm import tqdm
import pickle
import scipy.io
import helper
import os
import pandas as pd
import helper
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def concatenate_all(specific_file, args, type_concat):
	final_residuals = []
	for i in tqdm(range(args.total_batches)):

		file_path = args.to_save_path
		file_name = specific_file + "_residuals_part" + str(i) + "of" + str(args.total_batches)
		if type_concat == 'rmse':
			file_path += "residuals_od32/"
		elif type_concat == 'predictions':
			file_path += "predictions_od32/"
			file_name += "-decoding-predictions"
		elif type_concat == 'rsa':
			file_path += "rsa/"
		elif type_concat == 'llh':
			file_path += "llh/"
			file_name += "-llh"
		elif type_concat == 'fdr':
			file_path += "fdr/"
		elif type_concat == 'ranking':
			file_path += "final_rankings/"
		elif type_concat == 'alpha':
			file_path += "alphas/"
		else:
			print("ERROR")
		
		part = pickle.load( open( file_path + file_name + ".p", "rb" ) )
		print("MAX FOR BATCH " + str(i) + ": " + str(np.max(part)))
		print("MIN FOR BATCH " + str(i) + ": " + str(np.min(part)))
		final_residuals.extend(part)
	print("FILE NAME: " + str( file_path + specific_file))
	return final_residuals

def main():
	argparser = argparse.ArgumentParser(description="calculate rankings for model-to-brain")
	argparser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=12)
	argparser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	argparser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	argparser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation", action='store_true', default=True)
	argparser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model", action='store_true', default=False)
	argparser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain", action='store_true', default=False)
	argparser.add_argument("-glove", "--glove", action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	argparser.add_argument("-word2vec", "--word2vec", action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("-bert", "--bert", action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("-rand_embed", "--rand_embed", action='store_true', default=False, help="True if initialize random embeddings, False if not")
	argparser.add_argument("-random",  "--random", action='store_true', default=False, help="True if add cross validation, False if not")
	argparser.add_argument("-permutation",  "--permutation", action='store_true', default=False, help="True if permutation, False if not")
	argparser.add_argument("-permutation_region", "--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	argparser.add_argument("-normalize", "--normalize",  action='store_true', default=False, help="True if add normalization across voxels, False if not")
	argparser.add_argument("-local", "--local",  action='store_true', default=False, help="True if local, False if not")
	argparser.add_argument("-log", "--log",  action='store_true', default=False, help="True if use log coordinates, False if not")
	argparser.add_argument("-rmse", "--rmse",  action='store_true', default=False, help="True if rmse, False if not")
	argparser.add_argument("-ranking", "--ranking",  action='store_true', default=False, help="True if ranking, False if not")
	argparser.add_argument("-fdr", "--fdr",  action='store_true', default=False, help="True if fdr, False if not")
	argparser.add_argument("-llh", "--llh",  action='store_true', default=False, help="True if llh, False if not")
	argparser.add_argument("-rsa", "--rsa",  action='store_true', default=False, help="True if rsa, False if not")
	argparser.add_argument("-alpha", "--alpha",  action='store_true', default=False, help="True if alpha, False if not")
	argparser.add_argument("-total_batches", "--total_batches", type=int, help="total number of batches residual_name is spread across", default=100)
	
	### UPDATE FILE PATHS HERE ###
	argparser.add_argument("--fmri_path", default="/n/shieber_lab/Lab/users/cjou/fmri/", type=str, help="file path to fMRI data on the Odyssey cluster")
	argparser.add_argument("--to_save_path", default="/n/shieber_lab/Lab/users/cjou/", type=str, help="file path to and create rmse/ranking/llh on the Odyssey cluster")
	### UPDATE FILE PATHS HERE ###

	args = argparser.parse_args()

	# check conditions // can remove when making pipeline
	if args.brain_to_model and args.model_to_brain:
		print("select only one flag for brain_to_model or model_to_brain")
		exit()
	if (not args.brain_to_model and not args.model_to_brain) and not args.rsa:
		print("select at least flag for brain_to_model or model_to_brain // or rsa")
		exit()
	# if not args.rmse and not args.ranking and not args.fdr and not args.llh and not args.rsa:
	# 	print("select at least flag for rmse, ranking, fdr, llh, rsa")
	# 	exit()

	print("getting volmask...")

	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

	print("CROSS VALIDATION: " + str(args.cross_validation))
	print("BRAIN_TO_MODEL: " + str(args.brain_to_model))
	print("MODEL_TO_BRAIN: " + str(args.model_to_brain))
	print("GLOVE: " + str(args.glove))
	print("WORD2VEC: " + str(args.word2vec))
	print("BERT: " + str(args.bert))
	print("RANDOM BRAIN: " + str(args.random))
	print("RANDOM EMBEDDINGS: " + str(args.rand_embed))
	print("PERMUTATION: " + str(args.permutation))
	print("PERMUTATION REGION: " + str(args.permutation_region))

	if args.local:
		volmask = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/volmask.p", "rb" ) )
		if args.ranking:
			atlas_vals = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/atlas_vals.p", "rb" ) )
			atlas_labels = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/atlas_labels.p", "rb" ) )
			roi_vals = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/roi_vals.p", "rb" ) )
			roi_labels = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/roi_labels.p", "rb" ) )

	else:
		volmask = pickle.load( open( "{}subj{}/volmask.p".format(args,fmri_path, args.subject_number), "rb" ) )
		if args.ranking:
			atlas_vals = pickle.load( open( "{}subj{}/atlas_vals.p".format(args,fmri_path, args.subject_number), "rb" ) )
			atlas_labels = pickle.load( open( "{}subj{}/atlas_labels.p".format(args,fmri_path, args.subject_number), "rb" ) )
			roi_vals = pickle.load( open( "{}subj{}/roi_vals.p".format(args,fmri_path, args.subject_number), "rb" ) )
			roi_labels = pickle.load( open( "{}subj{}/roi_labels.p".format(args,fmri_path, args.subject_number), "rb" ) )
	
	
	### MAKE PATHS ###
	print("making paths...")
	if not os.path.exists('../mat/'):
		os.makedirs('../mat/')

	if args.brain_to_model:
		metrics = ["rmse", "llh"]
	elif args.rsa:
		metrics = ["rsa"]
	elif args.alpha:
		metrics = ["alpha"]
	else:
		metrics = ["ranking", "rmse", "llh"]
	for layer in tqdm(range(1, args.num_layers+1)):
		print("LAYER: " + str(layer))
		for metric in metrics:
			print("METRIC: " + str(metric))
			if args.bert or args.word2vec or args.glove:
				specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-{}_layer{}"
				file_name = specific_file.format(
					args.subject_number,
					args.agg_type,
					layer
				)
			else:
				specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
				file_name = specific_file.format(
					args.subject_number,
					args.language,
					args.num_layers,
					args.model_type,
					layer,
					args.agg_type
				)

			print("transform coordinates...")
			if not args.word2vec and not args.glove and not args.bert and not args.random:
				specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
				file_format = specific_file.format(
					args.subject_number, 
					args.language, 
					args.num_layers, 
					args.model_type, 
					layer, 
					args.agg_type
				)
			else:
				file_format = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "-subj{}-{}_layer{}".format(args.subject_number, args.agg_type, layer)

			final_values = concatenate_all(file_format, args, metric)

			_ = helper.transform_coordinates(final_values, volmask, save_path="../mat/" + file_name, metric=metric)

	print('done.')

if __name__ == "__main__":
	main()