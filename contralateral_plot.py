import scipy.io
from tqdm import tqdm
import pickle
import numpy as np
import sys
import argparse
import os
import helper
import scipy.stats as stats

global a, b, c
def get_file(args):
	file = "../subj{}_{}.mat".format(
		args.subj_num,
		"volaal" if args.aal else "vollangloc"
		)
	contents = scipy.io.loadmat(file)
	if args.aal:
		return contents["volaal"]
	return contents["vollangloc"]

# def my_func(index):
# 	global a
# 	# print("INDEX")
# 	# print(a)
# 	# print(index)
# 	# print(a - index)
# 	# adsf
# 	return a - index

def contralateral(args):
	global a, b, c
	contents = get_file(args)
	a, b, c = contents.shape
	print("SHAPE: {} {} {}".format(a,b,c))
	contra = np.flip(contents, 0)
	# contra = np.apply_along_axis(my_func, 1, contents)

	file_name = "../contra_subj{}_{}.mat".format(
		args.subj_num,
		"volaal" if args.aal else "vollangloc"
		)
	scipy.io.savemat(file_name, dict(coords = contra))

def main():
	global a, b, c
	argparser = argparse.ArgumentParser(description="get contralateral coordinates")
	argparser.add_argument("-subj_num", '--subj_num', type=str, help="subject number", required=True)
	argparser.add_argument("-aal", '--aal', action='store_true', default=False, help="volaal")
	args = argparser.parse_args()

	contralateral(args)
	print("done.")

if __name__ == "__main__":
	main()