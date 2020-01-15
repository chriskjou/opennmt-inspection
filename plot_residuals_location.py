import numpy as np
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

def clean_roi(roi_vals, roi_labels):
	roi_vals = roi_vals.reshape((len(roi_vals), ))
	final_roi_labels = []
	for val_index in roi_vals:
		if val_index == 0:
			final_roi_labels.append("other")
		else:
			final_roi_labels.append(roi_labels[val_index-1][0][0])
	return final_roi_labels

def clean_atlas(atlas_vals, atlas_labels):
	at_vals = atlas_vals.reshape((len(atlas_vals), ))
	at_labels = []
	for val_index in at_vals:
		at_labels.append(atlas_labels[val_index-1][0][0])
	return at_labels

def get_location(df, atype, layer_num, names, activations):
    df_agg = df[df.agg_type == atype][df.layer == layer_num]
    indices = []
    for name in names:
        index = df_agg.index[df_agg['atlas_labels'] == name].tolist()
        indices += index
    all_activations = [activations[x] for x in indices]
    return np.nansum(all_activations), np.nansum(all_activations) // len(all_activations)
    print("SUM: ", np.nansum(all_activations))
    print("AVG: ", np.nansum(all_activations) // len(all_activations))

def compare_aggregations(df):
	# g = sns.catplot(x="roi_labels", y="residuals", data=df, hue="agg_type", kind="bar", height=7.5, aspect=1.5)
	# g.set_xticklabels(rotation=90)
	#plt.show()
	return

def plot_aggregations(df, args, file_name):
	all_residuals = list(df.residuals)
	g = sns.catplot(x="roi_labels", y="residuals", data=df, hue="layer", kind="bar", height=7.5, aspect=1.5)
	g.set_axis_labels("", "RMSE")
	g.set(ylim=(min(all_residuals), max(all_residuals)/1.75))
	plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language] + ", " + str(bm) + " " + str(cv))
	plt.show()
	return

def plot_atlas(df, args, file_name, zoom=False):
	if args.cross_validation:
		cv = "Cross Validation"
	else:
		cv = ""
	if args.brain_to_model:
		bm = "Brain-to-Model"
	else:
		bm = "Model-to-Brain"
	all_residuals = list(df.residuals)
	g = sns.catplot(x="atlas_labels", y="residuals", data=df, height=17.5, aspect=1.5)
	g.set_xticklabels(rotation=90)
	if zoom:
		g.set(ylim=(min(all_residuals), 0.5)) #5 * math.pow(10, -11)))
		file_name += "-zoom"
	else:
		g.set(ylim=(min(all_residuals), max(all_residuals)))
	g.set_axis_labels("RMSE", "")
	if not args.rand_embed and not args.word2vec and not args.glove and not args.bert:
		plt.title("RMSE in all Brain Regions for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language] + + ", " + str(bm) + " " + str(cv))
	elif args.word2vec:
		plt.title("RMSE in all Brain Regions for " + map_dict[args.agg_type] + " Aggregation of Word2Vec")
	elif args.glove:
		plt.title("RMSE in all Brain Regions for " + map_dict[args.agg_type] + " Aggregation of GLoVE")
	elif args.bert:
		plt.title("RMSE in all Brain Regions for " + map_dict[args.agg_type] + " Aggregation of BERT")	
	else: # args.rand_embed:
		plt.title("RMSE in all Brain Regions for " + map_dict[args.agg_type] + " Aggregation of Random Embeddings")	
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	# plt.show()
	return

def plot_roi(df, args, file_name, zoom=False):
	if args.cross_validation:
		cv = "Cross Validation"
	else:
		cv = ""
	if args.brain_to_model:
		bm = "Brain-to-Model"
	else:
		bm = "Model-to-Brain"
	all_residuals = list(df.residuals)
	g = sns.catplot(x="roi_labels", y="residuals", data=df, height=7.5, aspect=1.5)
	g.set_xticklabels(rotation=90)
	if zoom:
		print(min(all_residuals))
		g.set(ylim=(0, min(all_residuals) * 15)) #5 * math.pow(10, -11)))
		file_name += "-zoom"
	else:
		g.set(ylim=(min(all_residuals), max(all_residuals)))
	g.set_axis_labels("RMSE", "")
	if not args.rand_embed and not args.word2vec and not args.glove and not args.bert:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language])
	elif args.word2vec:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of Word2Vec" + ", " + str(bm) + " " + str(cv))
	elif args.glove:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of GLoVE" + ", " + str(bm) + " " + str(cv))
	elif args.bert:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of BERT" + ", " + str(bm) + " " + str(cv))	
	elif args.random and args.rand_embed:
		plt.title("RMSE in all Language Regions for Random Activations and Embeddings, " + str(bm) + " " + str(cv))	
	else: # args.rand_embed:
		plt.title("RMSE in all Language Regions for Random Embeddings, " + str(bm) + " " + str(cv))
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	return

def plot_boxplot_for_atlas(df, args, file_name):
	if args.cross_validation:
		cv = "Cross Validation"
	else:
		cv = ""
	if args.brain_to_model:
		bm = "Brain-to-Model"
	else:
		bm = "Model-to-Brain"
	all_residuals = list(df.residuals)
	g = sns.catplot(x="atlas_labels", y="residuals", data=df, height=17.5, aspect=1.5, kind="box")
	g.set_xticklabels(rotation=90)
	g.set(ylim=(min(all_residuals), max(all_residuals)))
	g.set_axis_labels("RMSE", "")
	if not args.rand_embed and not args.word2vec and not args.glove and not args.bert:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language])
	elif args.word2vec:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of Word2Vec" + ", " + str(bm) + " " + str(cv))
	elif args.glove:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of GLoVE" + ", " + str(bm) + " " + str(cv))
	elif args.bert:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of BERT" + ", " + str(bm) + " " + str(cv))	
	elif args.random and args.rand_embed:
		plt.title("RMSE in all Language Regions for Random Activations and Embeddings, " + str(bm) + " " + str(cv))	
	else: # args.rand_embed:
		plt.title("RMSE in all Language Regions for Random Embeddings, " + str(bm) + " " + str(cv))
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	return

def plot_boxplot_for_roi(df, args, file_name):
	if args.cross_validation:
		cv = "Cross Validation"
	else:
		cv = ""
	if args.brain_to_model:
		bm = "Brain-to-Model"
	else:
		bm = "Model-to-Brain"
	all_residuals = list(df.residuals)
	g = sns.catplot(x="roi_labels", y="residuals", data=df, height=7.5, aspect=1.5, kind="box")
	g.set_xticklabels(rotation=90)
	# g.set(ylim=(min(all_residuals), max(all_residuals)))
	g.set(ylim=(min(all_residuals), 50))
	g.set_axis_labels("RMSE", "")
	if not args.rand_embed and not args.word2vec and not args.glove and not args.bert:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language])
	elif args.word2vec:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of Word2Vec" + ", " + str(bm) + " " + str(cv))
	elif args.glove:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of GLoVE" + ", " + str(bm) + " " + str(cv))
	elif args.bert:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of BERT" + ", " + str(bm) + " " + str(cv))	
	elif args.random and args.rand_embed:
		plt.title("RMSE in all Language Regions for Random Activations and Embeddings, " + str(bm) + " " + str(cv))	
	else: # args.rand_embed:
		plt.title("RMSE in all Language Regions for Random Embeddings, " + str(bm) + " " + str(cv))
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	return

def plot_violinplot_for_atlas(df, args, file_name):
	plt.clf()
	if args.cross_validation:
		cv = "Cross Validation"
	else:
		cv = ""
	if args.brain_to_model:
		bm = "Brain-to-Model"
	else:
		bm = "Model-to-Brain"
	all_residuals = list(df.residuals)
	g = sns.violinplot(x="atlas_labels", y="residuals", data=df, height=17.5, aspect=1.5)
	g.set_xticklabels(rotation=90)
	# g.set(ylim=(min(all_residuals), max(all_residuals)))
	# g.set_axis_labels("RMSE", "")
	if not args.rand_embed and not args.word2vec and not args.glove and not args.bert:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language])
	elif args.word2vec:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of Word2Vec" + ", " + str(bm) + " " + str(cv))
	elif args.glove:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of GLoVE" + ", " + str(bm) + " " + str(cv))
	elif args.bert:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of BERT" + ", " + str(bm) + " " + str(cv))	
	else: # args.rand_embed:
		plt.title("RMSE in all Language Regions for Random Embeddings")	
	
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	return

def plot_violinplot_for_roi(df, args, file_name):
	plt.clf()
	if args.cross_validation:
		cv = "Cross Validation"
	else:
		cv = ""
	if args.brain_to_model:
		bm = "Brain-to-Model"
	else:
		bm = "Model-to-Brain"
	all_residuals = list(df.residuals)
	g = sns.violinplot(x="roi_labels", y="residuals", data=df, height=7.5, aspect=1.5)
	# g.set_xticklabels(rotation=90)
	g.set(ylim=(min(all_residuals), max(all_residuals)))
	# g.set_axis_labels("RMSE", "")
	if not args.rand_embed and not args.word2vec and not args.glove and not args.bert:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of " + str(args.which_layer) + "-Layer " + str(args.model_type).upper() + " English-to-" + map_dict[args.language])
	elif args.word2vec:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of Word2Vec" + ", " + str(bm) + " " + str(cv))
	elif args.glove:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of GLoVE" + ", " + str(bm) + " " + str(cv))
	elif args.bert:
		plt.title("RMSE in all Language Regions for " + map_dict[args.agg_type] + " Aggregation of BERT" + ", " + str(bm) + " " + str(cv))	
	elif args.random and args.rand_embed:
		plt.title("RMSE in all Language Regions for Random Activations and Embeddings, " + str(bm) + " " + str(cv))	
	else: # args.rand_embed:
		plt.title("RMSE in all Language Regions for Random Embeddings, " + str(bm) + " " + str(cv))
	
	plt.savefig("../visualizations/" + str(file_name) + ".png")
	return

def main():

	argparser = argparse.ArgumentParser(description="plot RMSE by location")
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
	argparser.add_argument("-random",  "--random", action='store_true', default=False, help="True if initialize random brain activations, False if not")
	argparser.add_argument("-rand_embed",  "--rand_embed", action='store_true', default=False, help="True if initialize random embeddings, False if not")
	argparser.add_argument("-bert",  "--bert", action='store_true', default=False, help="True if initialize bert embeddings, False if not")
	argparser.add_argument("-permutation",  "--permutation", action='store_true', default=False, help="True if permutation, False if not")
	argparser.add_argument("-permutation_region", "--permutation_region",  action='store_true', default=False, help="True if permutation by brain region, False if not")
	argparser.add_argument("-local",  "--local", action='store_true', default=False, help="True if running locally")
	argparser.add_argument("-hard_drive",  "--hard_drive", action='store_true', default=False, help="True if running from hard drive")
	args = argparser.parse_args()

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
	file_loc = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(bertlabel) + str(direction) + str(validate) + "subj{}_parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"

	file_name = file_loc.format(
		args.subject_number, 
		args.language, 
		args.num_layers, 
		args.model_type, 
		args.which_layer, 
		args.agg_type
	)

	residual_file = "../rmses/concatenated-" + str(file_name) + ".p"

	# file_name = residual_file.split("/")[-1].split(".")[0]
	all_residuals = pickle.load( open( residual_file, "rb" ) )

	# get atlas and roi
	if not args.local:
		atlas_vals = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/atlas_vals.p", "rb" ) )
		atlas_labels = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/atlas_labels.p", "rb" ) )
		roi_vals = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/roi_vals.p", "rb" ) )
		roi_labels = pickle.load( open( f"/n/shieber_lab/Lab/users/cjou/fmri/subj{args.subject_number}/roi_labels.p", "rb" ) )
	
	elif args.hard_drive:
		atlas_vals = pickle.load( open( f"/Volumes/passport/\!RESEARCH/examplesGLM/subj{args.subject_number}/atlas_vals.p", "rb" ) )
		atlas_labels = pickle.load( open( f"/Volumes/passport/\!RESEARCH/examplesGLM/subj{args.subject_number}/atlas_labels.p", "rb" ) )
		roi_vals = pickle.load( open( f"/Volumes/passport/\!RESEARCH/examplesGLM/subj{args.subject_number}/roi_vals.p", "rb" ) )
		roi_labels = pickle.load( open( f"/Volumes/passport/\!RESEARCH/examplesGLM/subj{args.subject_number}/roi_labels.p", "rb" ) )
		
	else:
		atlas_vals = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/atlas_vals.p", "rb" ) )
		atlas_labels = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/atlas_labels.p", "rb" ) )
		roi_vals = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/roi_vals.p", "rb" ) )
		roi_labels = pickle.load( open( f"../examplesGLM/subj{args.subject_number}/roi_labels.p", "rb" ) )

	print("INITIAL:")
	print(len(atlas_vals))
	print(len(atlas_labels))
	print(len(roi_vals))
	print(len(roi_labels))

	final_roi_labels = clean_roi(roi_vals, roi_labels)
	at_labels = clean_atlas(atlas_vals, atlas_labels)

	print("CLEANING")
	print(len(final_roi_labels))
	print(len(at_labels))

	if not os.path.exists('../visualizations/'):
		os.makedirs('../visualizations/')

	# make dataframe
	print(len(list(range(len(all_residuals)))))
	print(len(all_residuals))
	print(len(at_labels))
	print(len(final_roi_labels))

	df_dict = {'voxel_index': list(range(len(all_residuals))),
			'residuals': all_residuals,
			'atlas_labels': at_labels,
			'roi_labels': final_roi_labels}

	df = pd.DataFrame(df_dict)

	# create plots
	print("creating plots...")
	# plot_roi(df, args, file_name + "-roi", zoom=False)
	# plot_atlas(df, args, file_name + "-atlas", zoom=False)
	# plot_roi(df, args, file_name + "-roi", zoom=True)
	# plot_atlas(df, args, file_name + "-atlas", zoom=True)
	plot_boxplot_for_roi(df, args, file_name + "-boxplot-roi")
	# plot_boxplot_for_atlas(df, args, file_name + "-boxplot-atlas")
	# plot_violinplot_for_roi(df, args, file_name + "-violinplot-roi")
	# plot_violinplot_for_atlas(df, args, file_name + "-violinplot-atlas")
	# plot_aggregations(df, args, file_name + "-agg")

	print("done.")

	return

if __name__ == "__main__":
    main()
