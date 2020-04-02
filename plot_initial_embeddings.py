import numpy as np
import scipy.io
import pickle
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
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
	if not args.glove and not args.word2vec and not args.bert and not args.random:
		plt.title("Embeddings for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language])
	if args.glove:
		plt.title("Glove Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	if args.word2vec:
		plt.title("Word2Vec Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	if args.bert:
		plt.title("BERT Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	if args.random:
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
	if not args.glove and not args.word2vec and not args.bert and not args.random:
		plt.title("Embeddings for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language])
	if args.glove:
		plt.title("Glove Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	if args.word2vec:
		plt.title("Word2Vec Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	if args.bert:
		plt.title("BERT Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	if args.random:
		plt.title("Random Embeddings for " + map_dict[args.agg_type] + " Aggregation")
	# plt.show()
	plt.savefig("../visualizations/" + str(file_name) + "-boxplot.png")
	return

def plot_agg(df, file_name, hue):
	plt.clf()
	plt.figure(figsize=(60, 8))
	ax = sns.boxplot(x="dimension", y="embeddings", hue=hue, data=df, palette="Set3")
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
	# ax = sns.catplot(x="dimension", y="embeddings", hue="agg_type", col="baseline", data=df, kind="box", palette="Set3")
	# plt.show()
	plt.savefig(file_name)
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
	argparser.add_argument("-random", "--random", action='store_true', default=False, help="True if initialize random embeddings, False if not")
	argparser.add_argument("-opennmt", "--opennmt", action='store_true', default=False, help="True if plot opennmt embeddings, False if not")
	argparser.add_argument("-baseline", "--baseline", action='store_true', default=False, help="True if plot baseline embeddings, False if not")
	# argparser.add_argument("-subject_number", "--subject_number", type=int, default=1, help="subject number (fMRI data) for decoding")
	args = argparser.parse_args()

	print("getting arguments...")
	if args.baseline:
		df_arr = []
		df_label = []
		df_agg = []
		by_dimension = []
		by_dimension_label = []
		by_dimension_type = []
		by_dimension_agg = []

		for agg_type in ["avg", "max", "min", "last"]:
			glove_file_loc = f"../embeddings/glove/{agg_type}.p"
			w2v_file_loc = f"../embeddings/word2vec/{agg_type}.p"
			glove = np.array(pickle.load( open( glove_file_loc, "rb" ) ))
			word2vec = np.array(pickle.load( open( w2v_file_loc, "rb" ) ))

			# by dimension
			dims = list(range(1, glove.shape[1] + 1)) 
			dims_all = np.repeat(dims, glove.shape[0]) 
			dims_all_labels = list(dims_all) + list(dims_all)
			total_glove = glove.shape[0] * glove.shape[1]
			total_word2vec = word2vec.shape[0] * word2vec.shape[1]
			by_dimension_label.extend(dims_all_labels)

			g_by_col = glove.flatten(order='F')
			w_by_row = word2vec.flatten(order='F')
			by_dimension.extend(g_by_col)
			by_dimension.extend(w_by_row)
			by_dimension_type.extend(["glove"] * total_glove + ["word2vec"] * total_word2vec)
			by_dimension_agg.extend([agg_type] * (total_glove + total_word2vec))
			print([agg_type])

			# all dimensions
			glove = glove.flatten()
			word2vec = word2vec.flatten()

			label_types = ["glove"] * len(glove) + ["word2vec"] * len(word2vec)

			df_label.extend(label_types)
			df_arr.extend(glove)
			df_arr.extend(word2vec)

			agg_types = [agg_type] * (len(glove) + len(word2vec))
			df_agg.extend(agg_types)

		print("FOR DIMENSION: " + str(len(by_dimension)))
		print("FOR DIMENSION LABEL: " + str(len(by_dimension_label)))
		print("FOR DIMENSION AGG: " + str(len(by_dimension_agg)))

		df = pd.DataFrame({
			"dimension": by_dimension_label,
			"embeddings": by_dimension,
			"agg_type": by_dimension_agg,
			"baseline": by_dimension_type
		})

		df.head()

		avg_df = df.loc[df["agg_type"] == "avg"]
		min_df = df.loc[df["agg_type"] == "min"]
		max_df = df.loc[df["agg_type"] == "max"]
		last_df = df.loc[df["agg_type"] == "last"]

		plot_agg(avg_df, "../initial_embeddings_avg_baseline.png", "baseline")
		plot_agg(min_df, "../initial_embeddings_min_baseline.png", "baseline")
		plot_agg(max_df, "../initial_embeddings_max_baseline.png", "baseline")
		plot_agg(last_df, "../initial_embeddings_last_baseline.png", "baseline")

		print("ELNGTH ARR: " + str(len(df_arr)))
		print("LENGTH LABEL: " + str(len(df_label)))
		print("AGG: " + str(len(df_agg)))

		# for all
		df = pd.DataFrame({
			"baseline": df_label,
			"embeddings": df_arr,
			"agg_type": df_agg
		})

		df.head()

		plt.clf()
		plt.figure(figsize=(11, 8))
		# ax = sns.boxplot(x="agg_type", y="embeddings", hue="baseline", data=df, palette="Set3")
		ax = sns.catplot(x="baseline", y="embeddings", hue="agg_type", data=df, kind="violin", palette="Set3")
		# plt.show()
		plt.savefig("../initial_embeddings_across_agg_type.png")
	elif args.bert:
		df_arr = []
		df_agg = []
		by_dimension = []
		by_dimension_label = []
		by_dimension_type = []
		by_dimension_agg = []
		num_layers = 12
		df_layers = []

		for agg_type in ["avg", "max", "min", "last"]:
			for layer in range(1, num_layers+1):
				file_name = f"../embeddings/bert/layer{layer}/{agg_type}.p"
				bert = np.array(pickle.load( open( file_name, "rb" ) ))

				# by dimension
				dims = list(range(1, bert.shape[1] + 1)) 
				dims_all = np.repeat(dims, bert.shape[0]) 
				total_bert = bert.shape[0] * bert.shape[1]
				by_dimension_label.extend(dims_all)

				bert_by_col = bert.flatten(order='F')
				by_dimension.extend(bert_by_col)
				by_dimension_agg.extend([agg_type] * (total_bert))

				# all dimensions
				bert = bert.flatten()
				df_arr.extend(bert)

				layer_types = [layer] * total_bert
				df_layers.extend(layer_types)

			agg_types = [agg_type] * (total_bert * num_layers)
			df_agg.extend(agg_types)

		print("FOR DIMENSION: " + str(len(by_dimension)))
		print("FOR DIMENSION LABEL: " + str(len(by_dimension_label)))
		print("FOR DIMENSION AGG: " + str(len(by_dimension_agg)))

		df = pd.DataFrame({
			"dimension": by_dimension_label,
			"embeddings": by_dimension,
			"agg_type": by_dimension_agg
		})

		# df.head()

		avg_df = df.loc[df["agg_type"] == "avg"]
		min_df = df.loc[df["agg_type"] == "min"]
		max_df = df.loc[df["agg_type"] == "max"]
		last_df = df.loc[df["agg_type"] == "last"]

		plot_agg(avg_df, "../initial_embeddings_bert_avg_baseline.png", "agg_type")
		plot_agg(min_df, "../initial_embeddings_bert_min_baseline.png", "agg_type")
		plot_agg(max_df, "../initial_embeddings_bert_max_baseline.png", "agg_type")
		plot_agg(last_df, "../initial_embeddings_bert_last_baseline.png", "agg_type")

		print("ELNGTH ARR: " + str(len(df_arr)))
		print("LENGTH LAYER: " + str(len(df_layers)))
		print("AGG: " + str(len(df_agg)))

		# for all
		df = pd.DataFrame({
			"layer": df_layers,
			"embeddings": df_arr,
			"agg_type": df_agg
		})

		df.head()

		plt.clf()
		plt.figure(figsize=(40, 10))
		# ax = sns.boxplot(x="agg_type", y="embeddings", hue="baseline", data=df, palette="Set3")
		ax = sns.boxplot(x="layer", y="embeddings", hue="agg_type", data=df, palette="Set3")
		# plt.show()
		plt.legend(loc='lower left')
		plt.savefig("../initial_embeddings_bert_across_agg_type.png")
		pass
	elif args.opennmt:
		df_arr = []
		df_agg = []
		by_dimension = []
		by_dimension_label = []
		by_dimension_type = []
		by_dimension_agg = []
		num_layers = 4
		df_layers = []
		model_type = "brnn"

		for agg_type in ["avg", "max", "min", "last"]:
			for layer in range(1, num_layers+1):
				for language in ["spanish", "french", "german", "swedish", "italian"]:
					file_loc = "../embeddings/parallel/{0}/{1}layer-{2}/{3}/parallel-english-to-{0}-model-{1}layer-{2}-pred-layer{4}-{3}.mat"
					embed_loc = file_loc.format(
						language, 
						num_layers, 
						model_type, 
						agg_type,
						layer
					)
					print(embed_loc)
					file_name = embed_loc.split("/")[-1].split(".")[0]
					embedding = scipy.io.loadmat(embed_loc)
					embed_matrix = get_embed_matrix(embedding)
					print(embed_matrix.shape)

					# by dimension
					dims = list(range(1, embed_matrix.shape[1] + 1)) 
					dims_all = np.repeat(dims, embed_matrix.shape[0]) 
					total_nmt = embed_matrix.shape[0] * embed_matrix.shape[1]
					by_dimension_label.extend(dims_all)

					nmt_by_col = embed_matrix.flatten(order='F')
					by_dimension.extend(nmt_by_col)
					by_dimension_agg.extend([agg_type] * (total_nmt))

					# all dimensions
					embed_matrix = embed_matrix.flatten()
					df_arr.extend(embed_matrix)

					layer_types = [layer] * total_nmt
					df_layers.extend(layer_types)

			agg_types = [agg_type] * (total_nmt * num_layers)
			df_agg.extend(agg_types)

		print("FOR DIMENSION: " + str(len(by_dimension)))
		print("FOR DIMENSION LABEL: " + str(len(by_dimension_label)))
		print("FOR DIMENSION AGG: " + str(len(by_dimension_agg)))

		df = pd.DataFrame({
			"dimension": by_dimension_label,
			"embeddings": by_dimension,
			"agg_type": by_dimension_agg
		})

		# df.head()

		avg_df = df.loc[df["agg_type"] == "avg"]
		min_df = df.loc[df["agg_type"] == "min"]
		max_df = df.loc[df["agg_type"] == "max"]
		last_df = df.loc[df["agg_type"] == "last"]

		plot_agg(avg_df, "../initial_embeddings_bert_avg_baseline.png", "agg_type")
		plot_agg(min_df, "../initial_embeddings_bert_min_baseline.png", "agg_type")
		plot_agg(max_df, "../initial_embeddings_bert_max_baseline.png", "agg_type")
		plot_agg(last_df, "../initial_embeddings_bert_last_baseline.png", "agg_type")

		print("ELNGTH ARR: " + str(len(df_arr)))
		print("LENGTH LAYER: " + str(len(df_layers)))
		print("AGG: " + str(len(df_agg)))

		# for all
		df = pd.DataFrame({
			"layer": df_layers,
			"embeddings": df_arr,
			"agg_type": df_agg
		})

		df.head()

		plt.clf()
		plt.figure(figsize=(40, 10))
		# ax = sns.boxplot(x="agg_type", y="embeddings", hue="baseline", data=df, palette="Set3")
		ax = sns.boxplot(x="layer", y="embeddings", hue="agg_type", data=df, palette="Set3")
		# plt.show()
		plt.legend(loc='lower left')
		plt.savefig("../initial_embeddings_bert_across_agg_type.png")
		pass
	else:
		if not args.glove and not args.word2vec and not args.bert and not args.random:
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
		else:
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
				file_loc = f"../embeddings/rand_embed/rand_embed.p"
				file_name = f"rand_embed"
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
		print(file_name)
	print("done.")

	return

if __name__ == "__main__":
    main()
