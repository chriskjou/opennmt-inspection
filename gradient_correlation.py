import scipy.io
from tqdm import tqdm
import pickle
import numpy as np
import sys
import argparse
import os
import helper
import scipy.stats as stats

def remove_nan(x, y):
	indices = np.argwhere(~np.isnan(y))
	return x[indices], y[indices]

def calculate_correlation(args, x, y):
	# print("SHAPE CHECK")
	# print(x.shape)
	# print(y.shape)
	# print(np.sum(np.isinf(x)))
	# print(np.sum(np.isinf(y)))
	# print(np.sum(np.isnan(x)))
	# print(np.sum(np.isnan(y)))
	if args.argmax or args.slope:
		x,y = remove_nan(x,y)
		x = np.reshape(x, (len(x),))
		y = np.reshape(y, (len(y),))
	print("REMOVE CHECK")
	print(x.shape)
	print(y.shape)
	# print(np.sum(np.isinf(x)))
	# print(np.sum(np.isinf(y)))
	# print(np.sum(np.isnan(x)))
	# print(np.sum(np.isnan(y)))
	if args.spearman:
		rho, pval = stats.spearmanr(x, y)
		return rho, pval
	if args.pearson:
		rho, pval = stats.pearsonr(x, y)
		return rho, pval
	return "error"

def calculate_gradient(args, arr):
	coords = np.nonzero(arr)
	vals = np.transpose(arr[coords])
	# print(len(coords))
	# print(len(coords[0]))
	# print(np.array(coords).shape)
	# vals = np.reshape(vals, (1, len(vals)))
	coords = np.transpose(np.array(coords))
	# print("COORDS SHAPE: " + str(coords.shape))
	# print("VALS SHAPE: " + str(vals.shape))

	labels = ["x", "y", "z"]
	for d in [0,1,2]:
		df = np.transpose(coords[:, d])
		rho, pval = calculate_correlation(args, df, vals)
		print(str(labels[d]) + " DIRECTION")
		print("-- rho: " + str(rho))
		print("-- pval: " + str(pval))
		print()
	return


def main():
	argparser = argparse.ArgumentParser(description="gradient correlations")
	argparser.add_argument("-contra", '--contra', action='store_true', default=False, help="contra")
	argparser.add_argument("-spearman", '--spearman', action='store_true', default=False, help="spearman")
	argparser.add_argument("-pearson", '--pearson', action='store_true', default=False, help="pearson")
	argparser.add_argument("-slope", '--slope', action='store_true', default=False, help="slope")
	argparser.add_argument("-argmax", '--argmax', action='store_true', default=False, help="argmax")
	args = argparser.parse_args()

	rois = ["lmidposttemp", "lposttemp", "lmidanttemp", "lifg", "lanttemp", "lifgorb", "langg", "lmfg"]

	if args.spearman and args.pearson:
		print("error: select only one correlation")
		exit()
	if not args.spearman and not args.pearson:
		print("error: select only one correlation")
		exit()

	if args.slope and args.argmax:
		print("error: select only one argmax or slope")
		exit()

	if args.spearman and args.slope:
		print("error: use pearson correlation for slope")
		exit()
	if args.pearson and args.argmax:
		print("error: use spearman correlation for argmax")
		exit()

	print("opening files...")
	file_name = "bert_bms_sig"
	tag = "bert_sig"
	if args.contra:
		file_name = "contra_" + file_name
		tag = "contra_" + tag

	if args.argmax:
		file_name += "_argmax"
		tag += "_argmax"
	if args.slope:
		file_name += "_slope"
		tag += "_slope"

	file_contents = scipy.io.loadmat("../" + file_name + ".mat")

	print("for each region...")
	for roi in rois:
		print("ROI: " + str(roi))
		region = file_contents[tag][roi][0][0]
		# print(region.shape)
		calculate_gradient(args, region)
	print("done.")

if __name__ == "__main__":
	main()