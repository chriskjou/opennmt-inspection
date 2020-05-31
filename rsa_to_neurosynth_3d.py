import scipy.io
from tqdm import tqdm
import pickle
import argparse
import numpy as np

def main():
	parser = argparse.ArgumentParser("rsa correlations to 3d neurosynth regions")
	parser.add_argument("-num_layers", "--num_layers", help="Total number of layers", type=int, default=12)
	parser.add_argument("-subject_number", "--subject_number", help="fMRI subject number ([1:11])", type=int, default=1)
	parser.add_argument("-local",  "--local", action='store_true', default=True, help="True if running locally")
	
	labels = scipy.io.loadmat("../../projects/opennmt-inspection/neurosynth_labels.mat")["initial"] 
	
	print("done.")

if __name__ == "__main__":
	main()