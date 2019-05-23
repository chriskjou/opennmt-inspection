import numpy as np 
import pickle
import sys

def concatenate_all_residuals(residual_name, total_batches):
	final_residuals = []
	for i in range(total_batches):
		file_name = "../residuals/all_residuals_part" + str(i) + "of" + str(total_batches) + ".p"
		part = pickle.load( open( file_name, "rb" ) )
		final_residuals.extend(part)
	return final_residuals

def main():
	if len(sys.argv) != 3:
		print("usage: python get_residuals.py -residual_name -total_batches")
		exit()

	residual_name = sys.argv[1]
	total_batches = int(sys.argv[2])

	final_residuals = concatenate_all_residuals(residual_name, total_batches)

	file_name = "../residuals/concatenated_all_residuals.p"
	pickle.dump( final_residuals, open( file_name, "wb" ) )
	print("done.")
	return

if __name__ == "__main__":
    main()