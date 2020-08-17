import pickle
import argparse
import pandas as pd 
from tqdm import tqdm
import numpy as np 
import helper
import scipy.io
from scipy.stats import linregress, ttest_1samp, ttest_ind
from statsmodels.stats.anova import AnovaRM 
import matplotlib.pyplot as plt
import seaborn as sns
import helper
plt.switch_backend('agg')

bert_num_layers = 12
opennmt_num_layers = 4

bert = list(range(bert_num_layers))
baseline = [bert_num_layers, bert_num_layers+1]
opennmt = list(range(bert_num_layers+2, bert_num_layers+opennmt_num_layers+2))

def get_file_name(args, subj, layer, model):
	if args.local:
		file_loc = "../mat/"
	else:
		file_loc = "../../mat/"
	if model == "bert" or model == "glove" or model == "word2vec":
		file_name = "{}model2brain_cv_-subj{}-avg_layer{}_no_spotlight-llh-3dtransform-llh.mat".format(
			model, 
			subj,
			layer
		)
	else:
		file_name = "model2brain_cv_-subj{}-parallel-english-to-spanish-model-4layer-brnn-pred-layer{}-avg_no_spotlight-llh-3dtransform-llh.mat".format(
			subj,
			layer
		)
	return file_loc + file_name

def get_file_contents(file_name):
	vals = scipy.io.loadmat( open( file_name, "rb" ) )["metric"]
	return vals

def get_model_contents(args, df, subj, num_layers, model, common_space):
	for layer in list(range(1, num_layers+1)):
		file_name = get_file_name(args, subj, layer, model)
		file_contents = get_file_contents(file_name)
		voxel_values = file_contents[np.nonzero(common_space)]
		df.append(voxel_values)
	return df

def calculate_anova(df):
	pvals = []
	num_subjs = 9
	for vox in tqdm(df):
		vox = calculate_avg_across_models(vox)
		vox = np.append(vox, np.reshape(np.array(list(range(1, num_subjs+1))), (num_subjs,1)), 1)
		vox = pd.DataFrame(vox, columns=['bert','baseline','opennmt','subject'])
		sub_vox = vox.melt(id_vars=["subject"], 
				var_name="model", 
				value_name="corr")
		aovrm2way = AnovaRM(sub_vox, "corr", "model", within=["subject"])
		mod = aovrm2way.fit()
		pval = mod.summary().tables[0]["Pr > F"]["subject"]
		pvals.append(pval)
	return pvals

def calculate_avg_across_models(df):
	bert_num_layers = 12
	opennmt_num_layers = 4

	bert = list(range(bert_num_layers))
	baseline = [bert_num_layers, bert_num_layers+1]
	opennmt = list(range(bert_num_layers+2, bert_num_layers+opennmt_num_layers+2))
	vals = []

	# print("DF SHAPE: " + str(df.shape))
	for indices in [bert, baseline, opennmt]:
		region_vals = np.take(df, indices, axis=1) 
		avg_model_vals = np.mean(region_vals, axis=1)
		vals.append(avg_model_vals)
	return np.transpose(vals)

def plot_graphs(args, df, metric, file_name):
	df = pd.melt(df.reset_index(), id_vars=["index"], value_vars=df.columns, var_name='model', value_name='LLH') 
	plt.clf()
	sns.set(style="darkgrid")
	if args.aal:
		plt.figure(figsize=(24, 9))
		plt.xticks(rotation=90)
	else:
		plt.figure(figsize=(10, 9))
	g = sns.barplot(x="model", y="LLH", data=df, ci=68)
	plt.savefig("../../visualizations/" + str(file_name) + "_barplot_" + str(metric) + ".png", bbox_inches='tight')
	return

def calculate_avg_counts_across_models(df):
	vals = []
	for indices in [bert, baseline, opennmt]:
		region_vals = np.take(df, indices, axis=1) 
		avg_model_vals = np.mean(region_vals, axis=1)
		vals.append(avg_model_vals)
	return np.transpose(vals)

def get_counts(df):
	dims = df.shape
	counts = []
	for vox in tqdm(df):
		vox_counts = np.zeros((dims[1], dims[2]))
		max_per_row = np.argmax(vox, axis=1) 
		for elem in range(len(max_per_row)):
			which_index = max_per_row[elem]
			vox_counts[elem][which_index] = 1
		counts.append(vox_counts)
	return np.sum(np.array(counts), axis=0)

def plot_count_graphs(args, df, metric, file_name):
	count_vals = get_counts(df)
	# print("GETTING COUNTS: " + str(df.shape))
	df = calculate_avg_counts_across_models(count_vals)
	# print("DF REGION AVG: " + str(df.shape))
	df = pd.DataFrame(df, columns=['bert','baseline','opennmt'])
	# print(df.head())
	df = pd.melt(df.reset_index(), id_vars=["index"], value_vars=df.columns, var_name='model', value_name='count') 
	# print(df.head())
	plt.clf()
	sns.set(style="darkgrid")
	if args.aal:
		plt.figure(figsize=(24, 9))
		plt.xticks(rotation=90)
	else:
		plt.figure(figsize=(10, 9))
	g = sns.barplot(x="model", y="count", data=df, ci=68)
	plt.savefig("../../visualizations/" + str(file_name) + "_count_barplot_" + str(metric) + ".png", bbox_inches='tight')
	return

def main():
	parser = argparse.ArgumentParser("calculate nested cv model significance")
	parser.add_argument("-count", "--count", action='store_true', default=False, help="use counter")
	parser.add_argument("-aal",  "--aal", action='store_true', default=False, help="True if use RSA aal")
	parser.add_argument("-local",  "--local", action='store_true', default=False, help="True if local")
	parser.add_argument("-use_cache",  "--use_cache", action='store_true', default=False, help="True if use cache pval")
	parser.add_argument("-avg",  "--avg", action='store_true', default=False, help="True if avg")
	args = parser.parse_args()

	subjects = [1,2,4,5,7,8,9,10,11]
	print("finding common space...")
	common_space = helper.load_common_space(subjects, local=args.local)
	voxel_coordinates = np.transpose(np.nonzero(common_space))
	print("COMMON SPACE: " + str(common_space.shape))
	print("VOXEL COORDINATES: " + str(voxel_coordinates.shape))
	
	volmask, num_regions, labels, vals, file_name = helper.get_voxel_labels(args)
	
	vals_3d = helper.convert_np_to_matlab(vals, volmask)
	labels_vals = vals_3d[np.nonzero(common_space)]

	# get values
	print("getting values...")
	bert_num_layers = 12
	opennmt_num_layers = 4

	df_full = []
	for subj in subjects:
		df_subj = []
		df_subj = get_model_contents(args, df_subj, subj, bert_num_layers, "bert", common_space)
		df_subj = get_model_contents(args, df_subj, subj, 1, "glove", common_space)
		df_subj = get_model_contents(args, df_subj, subj, 1, "word2vec", common_space)
		df_subj = get_model_contents(args, df_subj, subj, opennmt_num_layers, "opennmt", common_space)
		df_full.append(np.transpose(df_subj))

	df_full = np.stack(df_full, axis=1)
	print("DF FULL SHAPE: " + str(df_full.shape))

	print("calculate significant voxels...")
	if args.use_cache:
		sig_pvals = pickle.load( open( "../sig_pvals.p", "rb") )
	else:
		sig_pvals = calculate_anova(df_full)
		pickle.dump( sig_pvals, open("../sig_pvals.p", "wb" ) )

	sig_pvals_05 = (np.array(sig_pvals) < 0.05).astype(bool)

	print("aggregating...")
	# get values per region
	for region in tqdm(range(1, num_regions + 1)):
		indices_bool = (labels_vals == region).astype(bool)
		# print("INDICES SHAPE: " + str(indices_bool.shape))
		# print("NUM IN REGION: " + str(np.sum(indices_bool)))
		sig_indices_bol = np.array(indices_bool) & np.array(sig_pvals_05)
		# print("SIG INDICES SHAPE: " + str(sig_indices_bol.shape))
		# print("SiG NUM IN REGION: " + str(np.sum(sig_indices_bol)))
		indices = np.where(sig_indices_bol == True)[0]
		region_vals = np.take(df_full, indices, axis=0) 
		# print("DF REGION: " + str(region_vals.shape))
		if args.avg:
			avg_region_vals = np.mean(region_vals, axis=0)
			# print("DF REGION AVG: " + str(avg_region_vals.shape))
			avg_region_df = calculate_avg_across_models(avg_region_vals)
			# print("AVERAGE MODELS: " + str(np.array(avg_region_df).shape))
			df = pd.DataFrame(-avg_region_df, columns=['bert','baseline','opennmt'])
			df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
			plot_graphs(args, df, "llh", file_name + str(labels[region-1]))
		if args.count:
			plot_count_graphs(args, region_vals, "llh", file_name + str(labels[region-1]))

	print("done.")

if __name__ == "__main__":
	main()