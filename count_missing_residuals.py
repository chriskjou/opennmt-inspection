import glob
import os
import sys

def print_lst(lst):
    for x in lst:
        print(x)
    print(len(lst))

def main():
    if len(sys.argv) != 2:
        print('Usage: python count_missing_residuals.py -residual_stub')
        print('Example: python count_missing_residuals.py parallel-english-to-spanish-model-2layer-brnn-pred')

    residual_stub = sys.argv[1]
    filename_lst = glob.glob("../residuals/"+residual_stub+"*.p")
    filename_lst = [x.split("/")[-1] for x in filename_lst]
    print_lst(filename_lst)
    desired_filenames = []
    for layer_num in range(1, 3):
        for agg_type in ["min", "max", "avg", "last"]:
            for num in range(100):
                desired_filenames.append(f'parallel-english-to-spanish-model-2layer-brnn-pred-layer{layer_num}-{agg_type}_residuals_part{num}of100.p')
    missing_filenames = [x for x in desired_filenames if x not in filename_lst]
    print_lst(missing_filenames)

if __name__ == "__main__": main()
