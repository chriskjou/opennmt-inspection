import numpy as np 
import pickle
import sys

def concatenate_all_residuals(residual_name, language, num_layers, model_type, layer, agg_type, total_batches):
	final_residuals = []
	for i in range(total_batches):
		specific_file = "parallel-english-to-" + str(language) + "-model-" + str(num_layers) + "layer-" + str(model_type) + "-pred-layer" + str(layer) + "-" + str(agg_type)
		file_name = "../residuals/" + specific_file + "_residuals_part" + str(i) + "of" + str(total_batches) + ".p"
		part = pickle.load( open( file_name, "rb" ) )
		final_residuals.extend(part)
	return final_residuals

def main():
	if len(sys.argv) != 3:
		print("usage: python get_residuals.py -residual_name -total_batches")
		exit()

	languages = 'spanish' #['spanish', 'german', 'italian', 'french', 'swedish']
	num_layers = 2 #[2, 4]
	model_type = 'brnn' #['brnn', 'rnn']
	agg_type = ['avg', 'max', 'min', 'last']
	subj_num = 1
	nbatches = 100

	residual_name = sys.argv[1]
	total_batches = int(sys.argv[2])

	for atype in agg_type:
		for layer in list(range(1,num_layers+1)):
			final_residuals = concatenate_all_residuals(residual_name, languages, num_layers, model_type, layer, atype, total_batches)
			specific_file = "parallel-english-to-" + str(languages) + "-model-" + str(num_layers) + "layer-" + str(model_type) + "-pred-layer" + str(layer) + "-" + str(atype)
			file_name = "../final_residuals/concatenated-all-residuals-" + str(specific_file) + ".p"
			pickle.dump( final_residuals, open( file_name, "wb" ) )
	print("done.")
	return

if __name__ == "__main__":
    main()