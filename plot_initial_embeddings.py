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

map_dict = {
	'avg': "Average",
	'min': "Minimum", 
	'max': "Maximum",
	'last': "Last",
	"spanish": "Spanish",
	"swedish": "Swedish",
	"french": "French",
	"german": "German",
	"italian": "Italian"
}

def get_embed_matrix(embedding):
	dict_keys = list(embedding.keys())[3:]
	embed_matrix = np.array([embedding[i][0][1:] for i in dict_keys])
	in_training_bools = np.array([embedding[i][0][0] for i in dict_keys])
	return embed_matrix

def plot_graph(df, args, file_name):
	embed = df["embeddings"]
	plt.clf()
	g = sns.swarmplot(x=embed)
	# g.set_xticklabels(rotation=90)
	g.set(xlim=(min(embed) - 0.1, max(embed) + 0.1))
	# g.set_axis_labels("embeddings", "")
	if not args.glove and not args.word2vec and not args.bert and not args.rand_embed:
		plt.title("Embeddings for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language])
	if args.glove:
		plt.title("Glove Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	if args.word2vec:
		plt.title("Word2Vec Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	if args.bert:
		plt.title("BERT Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	if args.rand_embed:
		plt.title("Random Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	# plt.show()
	return

def plot_boxplot(df, args, file_name):
	embed = df["embeddings"]
	plt.clf()
	g = sns.boxplot(x=embed)
	# g.set_xticklabels(rotation=90)
	g.set(xlim=(min(embed) - 0.1, max(embed) + 0.1))
	# g.set_axis_labels("embeddings", "")
	if not args.glove and not args.word2vec and not args.bert and not args.rand_embed:
		plt.title("Embeddings for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language])
	if args.glove:
		plt.title("Glove Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	if args.word2vec:
		plt.title("Word2Vec Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	if args.bert:
		plt.title("BERT Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	if args.rand_embed:
		plt.title("Random Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	# plt.show()
	plt.savefig("../visualizations/" + str(file_name) + "-boxplot.png")
	return

def main():

	argparser = argparse.ArgumentParser(description="plot initial activations by location")
	# argparser.add_argument('-embedding_layer', type=str, help="Location of NN embedding (for a layer)", required=True)
	argparser.add_argument("-language", "--language", help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str, default='spanish')
	argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, default=2)
	argparser.add_argument("-model_type", "--model_type", help="Type of model ('brnn', 'rnn')", type=str, default='brnn')
	argparser.add_argument("-which_layer", "--which_layer", help="Layer of interest in [1: total number of layers]", type=int, default=1)
	argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str, default='avg')
	argparser.add_argument("-glove", "--glove", action='store_true', default=False, help="True if initialize glove embeddings, False if not")
	argparser.add_argument("-word2vec", "--word2vec", action='store_true', default=False, help="True if initialize word2vec embeddings, False if not")
	argparser.add_argument("-bert", "--bert", action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("-rand_embed", "--rand_embed", action='store_true', default=False, help="True if initialize random embeddings, False if not")
	# argparser.add_argument("-subject_number", "--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	args = argparser.parse_args()

	print("getting arguments...")
	if not args.glove and not args.word2vec and not args.bert:
		file_loc = "../embeddings/parallel/{0}/{1}layer-{2}/{3}/parallel-english-to-{0}-model-{1}layer-{2}-pred-layer{4}-{3}.mat"
		embed_loc = file_loc.format(
			args.language, 
			args.num_layers, 
			args.model_type, 
			args.agg_type,
			args.which_layer
		)
		print(embed_loc)

		print("getting embeddings...")
		# embed_loc = args.embedding_layer
		file_name = embed_loc.split("/")[-1].split(".")[0]
		embedding = scipy.io.loadmat(embed_loc)
		embed_matrix = get_embed_matrix(embedding)
		print(embed_matrix.shape)
	if args.glove:
		file_loc = f"../embeddings/glove/{args.agg_type}.p"
		file_name = f"glove-{args.agg_type}"
	if args.word2vec:
		file_loc = f"../embeddings/word2vec/{args.agg_type}.p"
		file_name = f"word2vec-{args.agg_type}"
	if args.bert:
		file_loc = f"../embeddings/bert/{args.agg_type}.p"
		file_name = f"bert-{args.agg_type}"
	if args.random:
		file_loc = f"../embeddings/random/{args.agg_type}.p"
		file_name = f"random-{args.agg_type}"
	# file_name = file_loc.split("/")[-1].split(".")[0]
	print("getting embeddings...")
	embed_matrix = pickle.load( open( file_loc, "rb" ) )
	embed_matrix = np.array(embed_matrix)
	print(embed_matrix.shape)


	avg = np.nanmean(embed_matrix, axis=0)
	print("EMBEDDINGS: " + str(np.shape(avg)))
	df = pd.DataFrame({
		"embeddings": avg
	})
	# print(df.head())

	# create visualization folder
	if not os.path.exists('../visualizations/'):
		os.makedirs('../visualizations/')

	print("plotting graphs...")
	plot_graph(df, args, file_name)
	plot_boxplot(df, args, file_name)

	# per sentence
	## TODO:

	print("done.")

	return

if __name__ == "__main__":
    main()
