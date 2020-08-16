Project Organization
------------
Folders will be generated one directory back from this current github repository; to keep all data, dependent program, and code centralized, it would be best to put this repo within its own folder. For example, part of my current setup is as shown below in a folder called `research`. When running on the Odyssey cluster, make sure the paths are still relative to each other in the same way or directly change the paths in the code.

	└─ research
	    ├── opennmt-inspection      					<- current github repo
	    │   ├── scripts       							<- added temporary scripts for testing
	    │   ├── notebooks								<- added jupyter notebooks
	    │   └── ...   									<- all other folders were from original branch
	    ├── bspmview          							<- dependency for MATLAB plotting
	    ├── examplesGLM									<- fMRI data downloaded from Google Drive
	    │   ├── subj1       							<- subfolder for subj1
	    │   ├── subj2       
	    │   ├── subj3    
	    │   └── ...        
	    ├── embeddings              					<- folder of embeddings generated from code
	    │   ├── bert       								<- subfolder of generated embeddings
	    │   ├── glove       
	    │   ├── gpt2    
	    │   └── ...        
	    ├── ccnl-fmri              						<- dependency for MATLAB plotting
	    ├── Language-fMRI          						<- dependency for MATLAB plotting
	    ├── spm12          								<- dependency for MATLAB plotting
	    ├── VBA-toolbox         						<- dependency for BMS library
	    ├── visualizations          					<- folder of graphs generated from code
	    ├── decoding_scripts   							<- folder of generated scripts for Odyssey from code
	    │   ├── model1       
	    │   ├── model2 
	    │   └── ...   
	    ├── mat           								<- downloaded folder from Odyssey
	    ├── GoogleNews-vectors-negative300.bin.gz 		<- downloaded word2vec dependency
		├── glove.6B 									<- downloaded glove dependency
	    └── ...           

Important added files are briefly described here with a more thorough description provided on file.

	└─ opennmt-inpsection
	    ├── calculate_slope_maps.py      				<- 									
	    ├── find_best_likelihood.py						<- 
	    ├── flair_embeddings.py          				<- 
	    ├── format_for_subject.py						<- 
	    ├── get_pretrained_embeddings.py           		<- 
	    ├── helper.py 									<- 
		├── make_nested_scripts.py 						<- 
		├── make_scripts.py 							<- 
		├── metric_across_layers.py 					<- 
		├── nested_convert_np_to_matlab.py 				<- 
		├── nested_cv_significance.py 					<- 
		├── nested_decoding.py 							<- 
		├── nested_vba_toolbox.py 						<- 
		├── plot_initial_activations.py 				<- 
		├── plot_initial_embeddings.py 					<- 
		├── plot_residuals_locations.py 				<- 
		├── significance_threshold.py 					<- 
		├── significant_llh.py 							<- 
	    └── ...    

Set-up
------------
1. Getting the Embeddings
2. Cleaning the fMRI data

Do these things first before doing any analysis. If you want to skip step (1) Getting the Embeddings, download the `embeddings/` folder from the Google Drive. Step (2) is required to download and reformat the fMRI data from the original MATLAB file to readable python pickle files.

# Getting the Embeddings
Ensure that `cleaned_sentencesGLM.txt` is included in the `opennmt-inspection` repo. This was cleaned from original fMRI data provided in `examplesGLM` and processed with `clean_data.py`.

Embeddings from Flair (Huggingface), pretrained (word2vec and GloVe), and opennmt-py are used and described below. Average, maximum, minimum, and last aggregations are generated per sentence embedding. Embeddings generated also located on the [Google Drive]().

## Generating Pretrained (word2vec and GloVe) Embeddings

Download the word2vec and glove individual word embeedings.

Make sure you are within the `opennmt-inspection` directory and run the code below
```
python get_pretrained_embeddings.py
```

This should generate the folder `../embeddings/` and the respective subfolders `../embeddings/word2vec/` and `../embeddings/glove/`.

## Generating Flair Embeddings (for BERT, GPT2, etc.)

Download [Flair](https://github.com/flairNLP/flair).

Make sure you are within the `opennmt-inspection` directory and run the code below for all the model embeddings desired. 
```
python flair_embeddings.py -local -bert
```

The `-local` flag is added if instead desired to directly generate the embeddings on the Odyssey cluster since rsync might take a while.

Currently only BERT (12 layers), RoBERTa (12 layers), XLM (24 layers), and GPT2 (12 layers) are included. Other models that can be generated are found on [Huggingface's documentation](https://huggingface.co/transformers/pretrained_models.html).

## Generating OpenNMT-py Embeddings
The translation data can be found at [this link](http://www.statmt.org/wmt16/translation-task.html) to the Europarl v7 corpus. After downloading via a preferred method, the corpus must first be multiparallelized (so that sentence *i* in the Czech, English, Spanish, etc. corpus all refer to the same idea). For example, one might run:
```
python multiparallelize_text.py ../corpus-data/czech/europarl-v7-en-cs.txt ../corpus-data/czech/europarl-v7-cs-en.txt ../corpus-data/spanish/europarl-v7-en-es.txt ../corpus-data/spanish/europarl-v7-es-en.txt
```
After saving the corpus data into a directory called corpus-data, and sorting into czech, spanish, etc. subfolders. Validation data also needs to be produced, which can be simply created by:
```
./make_validation.sh
```
Modify the variable *validation_num_lines* in ```make_validation.sh``` to be at ~5000 for good training results (see **step 2**) if you train on a large enough corpus for this to be possible.

Next, we first create the NLP model based on the translation data, and then we save its embeddings on our prediction text. For example, we could choose to create a Spanish translation NLP model:
```
python preprocess.py -train_src ../multiparallelize/training/parallel_src-training.txt  -train_tgt -valid_src -valid_tgt -save ../multiparallelize
```
This produces, as per below, ```multiparallelize.train.pt, multiparallelize.val.pt,``` and ```multiparallelize.vocab.pt```. We then take these *.pt* files and train (using the **--separate_layers** flag).

For example, we can train a English to Spanish model on the cluster with GPUs as follows:

```
python train.py -data data/english-to-spanish -save_model small-english-to-spanish-model -gpu 0 -separate_layers
```

At each epoch, a training model is saved with reference to accuracy in file name. Then, we want to run `translate.py` on the sentences read by the participants on the final model trained in 13th epoch and use the sentences read in the fMRI experiment (`cleaned_sentencesGLM.txt`) as follows 

```
python translate.py -model ../final_models/english-to-spanish-model_acc_61.26_ppl_6.28_e13.pt -src cleaned_sentencesGLM.txt -output ../predictions/english-to-spanish-model-pred.txt -replace_unk -verbose -dump_layers ../predictions/english-to-spanish-model-pred.pt
```

Embeddings in each layer are dumped into a corresponding `XX-pred.pt` which we have specified with the -dump_layers flag.

## Generating Random Embeddings (not used in analyses)
If desired to generate random embeddings uniformly distributed within a range, make sure you are within the `opennmt-inspection` directory and run
```
python random_embeddings.py
```

# Cleaning the fMRI data
We need to download the brain fMRI scans (in this case, those captured when reading ```examplesGLM.txt```). The fMRI scans are found [here](https://drive.google.com/drive/folders/1dfwmC6F8FuXlz_3fu2Q1SiSsZR_BY8RP) (you can use [this link](https://github.com/circulosmeos/gdown.pl) to download from drive via curl, wget, etc. ) *Note in the codebase we only regress to subject 1's embeddings because of computational tractability, but this is easily amended* (in ```odyssey_decoding.py``` and ```make_scripts.py```)
If you want, you can skip the earlier steps and download the NLP model embeddings from [this link](https://drive.google.com/drive/folders/1LNdXXD-W8ebm8WD1oIMKSw6Nt9rqsuWQ).
If you have the embeddings already, we still need to convert the subjects' fMRI data (in *.mat* format) into a more readable *.p* format; run
```
python format_for_subject.py --subject_number [X1 X2 X3]
```
Where ```X``` is the number of the subject whose *.mat* file you intend to process; note you can process one or more subjects at a time by listing multiple subject numbers (default is just the first subject (subject 1)). Type
```
python format_for_subject.py --help
```
for more.

Analyses Overview
------------
1. Bayesian Model Comparison
2. RSA by region
3. RSA by spotlight

# Bayesian Model Comparison (BMS)

## Dependencies
[VBA toolbox](https://github.com/MBB-team/VBA-toolbox)

Create VBA scripts by running the following line of code within the current directory. Make sure that you have downloaded the VBA toolbox and the respective file path is set.
```
python make_vba_scripts.py -path ../VBA-toolbox/ -email ckjou@college.harvard.edu
```

This should generate a master script and dependent scripts (default 100) in the VBA-toolbox directory.

Move `compare_model_families.m` into the VBA-toolbox directory. Update the directory path of your Odyssey cluster in the file as well. This file is what the scripts will run.

To submit scripts to the cluster for VBA analysis, run the master script 
```
sbatch calculate_nested_significance.sh
```
within the VBA-toolbox directory.

# RSA by region

# RSA by spotlight

Extending the Analysis
------------
1. Gradient Correlation
2. Plotting in 3D Brain space

# Gradient Correlation
To calculating an anatomical gradient index (1) BMS VBA-toolbox argmax; (2) RSA slope; or (3) RSA argmax, run the following line of code for respective analyses. Add the flag for `-contra` to calculate the gradient index for the same analysis but on the contralateral side of the brain.
```
python gradient_correlation.py -spearman 			# VBA argmax
python gradient_correlation.py -pearson -slope 		# RSA slope
python gradient_correlation.py -spearman -argmax	# RSA argmax
```

# Plotting in 3D Brain space
Redirect to the `Language-fMRI` folder. Make sure dependencies of `spm12`, `ccnl-fmri`, and `bspmview` are on the MATLAB path.

Load the corresponding .mat file into MATLAB. Follow the directions under `save_vol.m` to plot values in 3D brain space and generate brain slice/surface rendering photos.

Note: Current plots are not entirely exact due to missing MNI coordinates. Values less than 1 are also often clipped; make sure the proper data values are being plotted and the range on the plot is not cut off.
