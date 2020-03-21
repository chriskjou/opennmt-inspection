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
import pandas as pd
# plt.switch_backend('agg')
import seaborn as sns
# %matplotlib inline

def calculate_pval(layer1, layer2):
    pval = stats.ttest_rel(layer1, layer2)
    return pval

def compare_layers(layer1, layer2):
    diff = layer1-layer2
    return diff

def get_file(args, file_name):
    if args.ranking:
        metric = "ranking"
        path = "../mat/"
    elif args.rmse:
        metric = "rmse"
        path = "../3d-brain/"
    elif args.llh:
        metric = "llh"
        path = ""
    elif args.fdr:
        metric = "fdr"
        path = "../fdr/"
    else:
        print("error: check for valid method of correlation")
    save_path = path + str(file_name) + "-3dtransform-" + str(metric)
    print("LOADING FILE: " + str(save_path) + ".mat")
    values = scipy.io.loadmat(save_path + ".mat")
    return values["metric"]

def generate_file_name(args, which_layer):
    direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = helper.generate_labels(args)

    if args.bert or args.word2vec or args.glove:
        specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(
            bertlabel) + str(direction) + str(validate) + "-subj{}-{}_layer{}"
        file_name = specific_file.format(
            args.subject_number,
            args.agg_type,
            which_layer
        )
    else:
        specific_file = str(plabel) + str(prlabel) + str(rlabel) + str(elabel) + str(glabel) + str(w2vlabel) + str(
            bertlabel) + str(direction) + str(
            validate) + "-subj{}-parallel-english-to-{}-model-{}layer-{}-pred-layer{}-{}"
        file_name = specific_file.format(
            args.subject_number,
            args.language,
            args.num_layers,
            args.model_type,
            which_layer,
            args.agg_type
        )
    return file_name

def main():
    argparser = argparse.ArgumentParser(description="layer and subject group level comparison")
    argparser.add_argument("-subject_number", "--subject_number", type=int, default=1,
                           help="subject number (fMRI data) for decoding")
    argparser.add_argument("-cross_validation", "--cross_validation", help="Add flag if add cross validation",
                           action='store_true', default=False)
    argparser.add_argument("-brain_to_model", "--brain_to_model", help="Add flag if regressing brain to model",
                           action='store_true', default=False)
    argparser.add_argument("-model_to_brain", "--model_to_brain", help="Add flag if regressing model to brain",
                           action='store_true', default=False)
    argparser.add_argument("-agg_type", "--agg_type", help="Aggregation type ('avg', 'max', 'min', 'last')", type=str,
                           default='avg')
    argparser.add_argument("-language", "--language",
                           help="Target language ('spanish', 'german', 'italian', 'french', 'swedish')", type=str,
                           default='spanish')
    argparser.add_argument("-num_layers", "--num_layers", help="Total number of layers ('2', '4')", type=int, required=True)
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

    if args.num_layers != 12 and args.bert:
        print("error: please ensure bert has 12 layers")
        exit()

    if args.num_layers != 12 and (args.word2vec or args.random or args.permutation or args.glove):
        print("error: please ensure baseline has 1 layer")
        exit()

    print("NUMBER OF LAYERS: " + str(args.num_layers))

    print("generating file names...")
    layer1_file_name = generate_file_name(args, args.layer1)
    layer2_file_name = generate_file_name(args, args.layer2)

    print("retrieving file contents...")
    layer1 = get_file(args, layer1_file_name)
    layer2 = get_file(args, layer2_file_name)

    print("evaluating layers...")
    diff = compare_layers(layer1, layer2)
    print("DIFF")
    print(np.sum(diff))

    # generate heatmap
    heatmap_differences = np.zeros((args.num_layers, args.num_layers))
    for l1 in list(range(1, args.num_layers + 1)):
        for l2 in list(range(l1, args.num_layers + 1)):
            print("generating file names...")
            layer1_file_name = generate_file_name(args, l1)
            layer2_file_name = generate_file_name(args, l2)

            print("retrieving file contents...")
            layer1 = get_file(args, layer1_file_name)
            layer2 = get_file(args, layer2_file_name)

            diff = compare_layers(layer1, layer2)
            heatmap_differences[l1-1][l2-1] = np.sum(np.abs(diff))
            heatmap_differences[l2-1][l1-1] = np.sum(np.abs(diff))

    print(heatmap_differences.shape)
    print(heatmap_differences)

    Index = ['layer' + str(i) for i in range(1, args.num_layers + 1)]
    df = pd.DataFrame(heatmap_differences, index=Index, columns=Index)
    # plt.pcolor(df)
    # plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    # plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    # plt.show()
    sns.heatmap(df)
    plt.show()

    # pval = calculate_pval(layer1, layer2)
    # print("pval")
    # print(pval)

    # save_file(args, diff, "difference_" + a)
    # save_file(args, pval, "pval_" + )
    print("done.")
    return

if __name__ == "__main__":
    main()