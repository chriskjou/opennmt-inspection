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

def main():
	argparser = argparse.ArgumentParser(description="convert np to matlab for nested")
	argparser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	argparser.add_argument("-glove", "--glove", action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	argparser.add_argument("-word2vec", "--word2vec", action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("-bert", "--bert", action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("-opennmt", "--opennmt", action='store_true', default=False, help="True if initialize opennmt embeddings, False if not")
	argparser.add_argument("-local", "--local", action='store_true', default=False, help="True if running local, False if not")
	argparser.add_argument("-rsa", "--rsa",  action='store_true', default=False, help="True if rsa, False if not")
	
	### UPDATE FILE PATHS HERE ###
	argparser.add_argument("--fmri_path", default="/n/shieber_lab/Lab/users/cjou/fmri/", type=str, help="file path to fMRI data on the Odyssey cluster")
	argparser.add_argument("--to_save_path", default="/n/shieber_lab/Lab/users/cjou/", type=str, help="file path to and create rmse/ranking/llh on the Odyssey cluster")
	### UPDATE FILE PATHS HERE ###

	args = argparser.parse_args()

	if not args.glove and not args.word2vec and not args.bert and not args.opennmt:
		print("select at least one model")
		exit()

	print("getting volmask...")
	if args.local:
		volmask = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/volmask.p", "rb" ) )
	else:
		volmask = pickle.load( open( "{}subj{}/volmask.p".format(args.fmri_path, args.subject_number), "rb" ) )
	
	### MAKE PATHS ###
	print("making paths...")
	if not os.path.exists('../mat/'):
		os.makedirs('../mat/')

	if args.bert:
		num_layers = 12
		model = "bert"
	if args.word2vec:
		num_layers = 1
		model = "word2vec"
	if args.glove:
		num_layers = 1
		model = "glove"
	if args.opennmt:
		num_layers = 4
		model = "opennmt"

	for layer in tqdm(range(1, num_layers+1)):
		print("LAYER: " + str(layer))
		if args.rsa:
			file_path = "{}rsa/".format(args.to_save_path)
		else:
			file_path = "{}nested_llh/".format(args.to_save_path)
		if args.bert or args.word2vec or args.glove:
			file_name = "{}model2brain_cv_-subj{}-avg_layer{}_no_spotlight-llh".format(
				model, 
				args.subject_number,
				layer
			)
		else:
			file_name = "model2brain_cv_-subj{}-parallel-english-to-spanish-model-4layer-brnn-pred-layer{}-avg_no_spotlight-llh".format(
				args.subject_number,
				layer
			)

		final_values = pickle.load( open( file_path + file_name + ".p", "rb" ) )

		print(file_path + file_name)

		_ = helper.transform_coordinates(final_values, volmask, save_path="../mat/" + file_name, metric="llh")

	print('done.')

if __name__ == "__main__":
	main()