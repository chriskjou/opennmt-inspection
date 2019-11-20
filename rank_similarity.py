import numpy as np
import scipy.io
import pickle
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import argparse
import os
from tqdm import tqdm
import math

def compare_similarities():
	return

def main():

	argparser = argparse.ArgumentParser(description="plot initial activations by location")
	# argparser.add_argument('-embedding_layer', type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	argparser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	argparser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	# argparser.add_argument("-subject_number", "--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	args = argparser.parse_args()

if __name__ == "__main__":
	main()