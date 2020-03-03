from tqdm import tqdm
import scipy.io
import pickle
import numpy as np
import sys
from sklearn.model_selection import KFold
import argparse
import os
from scipy.stats import pearsonr
from scipy import stats
from scipy.linalg import lstsq
import random
import math
import statsmodels.stats.multitest as smm
import matplotlib.pyplot as plt
import helper
plt.switch_backend('agg')

def calculate_pval(layer1, layer2):
    pval = stats.ttest_rel(layer1, layer2)
    return pval

def compare_layers(layer1, layer2):
    diff = layer1-layer2
    return diff

def get_file(args, file_name):
    save_path = "/n/shieber_lab/Lab/users/cjou/fdr/" + str(file_name) + "_subj" + str(args.subject_number)
    values = scipy.io.loadmat(save_path + "-3dtransform-" + str(metric) + ".mat")
    return values

def main():
    argparser = argparse.ArgumentParser(description="layer and subject group level comparison")
    argparser.add_argument("-embedding_layer", "--embedding_layer", type=str,
                           help="Location of NN embedding (for a layer)", required=True)
    argparser.add_argument("-subject_number", "--subject_number", type=int, default=1,
                           help="subject number (fMRI data) for decoding")
    argparser.add_argument("-random", "--random", action='store_true', default=False,
                           help="True if initialize random brain activations, False if not")
    argparser.add_argument("-rand_embed", "--rand_embed", action='store_true', default=False,
                           help="True if initialize random embeddings, False if not")
    argparser.add_argument("-glove", "--glove", action='store_true', default=False,
                           help="True if initialize glove embeddings, False if not")
    argparser.add_argument("-word2vec", "--word2vec", action='store_true', default=False,
                           help="True if initialize word2vec embeddings, False if not")
    argparser.add_argument("-bert", "--bert", action='store_true', default=False,
                           help="True if initialize bert embeddings, False if not")
    argparser.add_argument("-normalize", "--normalize", action='store_true', default=False,
                           help="True if add normalization across voxels, False if not")
    argparser.add_argument("-permutation", "--permutation", action='store_true', default=False,
                           help="True if permutation, False if not")
    argparser.add_argument("-permutation_region", "--permutation_region", action='store_true', default=False,
                           help="True if permutation by brain region, False if not")
    argparser.add_argument("-layer1", "--layer1", help="Layer of interest in [1: total number of layers]",
                           type=int, default=1)
    argparser.add_argument("-layer2", "--layer2", help="Layer of interest in [1: total number of layers]",
                           type=int, default=1)
    argparser.add_argument("-single_subject", "--single_subject", help="if single subject analysis",
                           action='store_true', default=False)
    argparser.add_argument("-group_level", "--group_level", help="if group level analysis", action='store_true',
                           default=False)
    argparser.add_argument("-searchlight", "--searchlight", help="if searchlight", action='store_true', default=False)
    argparser.add_argument("-fdr", "--fdr", help="if apply FDR", action='store_true', default=False)
    argparser.add_argument("-subjects", "--subjects", help="subject numbers", type=str, default="")
    argparser.add_argument("-llh", "--llh", action='store_true', default=False,
                           help="True if calculate likelihood, False if not")
    argparser.add_argument("-ranking", "--ranking", action='store_true', default=False,
                           help="True if calculate ranking, False if not")
    argparser.add_argument("-rmse", "--rmse", action='store_true', default=False,
                           help="True if calculate rmse, False if not")
    argparser.add_argument("-rsa", "--rsa", action='store_true', default=False,
                           help="True if calculate rmse, False if not")
    args = argparser.parse_args()

    if args.layer1 == args.layer2:
        print("error: please select different layers for layer1 and layer2")
        exit()

    if not args.glove and not args.word2vec and not args.bert and not args.rand_embed:
        embed_loc = args.embedding_layer
        # embed_loc = "/Users/christinejou/Documents/research/embeddings/parallel/spanish/2layer-brnn/avg/parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg.mat"
        file_name = embed_loc.split("/")[-1].split(".")[0]
        embedding = scipy.io.loadmat(embed_loc)
        embed_matrix = get_embed_matrix(embedding)
    else:
        embed_loc = args.embedding_layer
        file_name = embed_loc.split("/")[-1].split(".")[0].split("-")[-1] + "_layer" + str(
            args.which_layer)  # aggregation type + which layer
        embed_matrix = pickle.load(open(embed_loc, "rb"))
        if args.word2vec:
            file_name += "word2vec"
        elif args.glove:
            file_name += "glove"
        elif args.bert:
            file_name += "bert"
        else:
            file_name += "random"

    layer1 = get_file(args, file_name)
    layer2 = get_file(args, file_name)

    diff = compare_layers(layer1, layer2)
    print("DIFF")
    print(diff)
    pval = calculate_pval(layer1, layer2)
    print("pval")
    print(pval)

    # save_file(args, diff, "difference_" + a)
    # save_file(args, pval, "pval_" + )
    print("done.")
    return

if __name__ == "__main__":
    main()