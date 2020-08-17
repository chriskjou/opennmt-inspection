# Project Organization
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

# Set-up
------------
1. Getting the Embeddings
2. Cleaning the fMRI data
3. Odyssey Cluster

Do these things first before doing any analysis. If you want to skip step (1) Getting the Embeddings, download the `embeddings/` folder from the Google Drive [here](https://drive.google.com/drive/folders/1LNdXXD-W8ebm8WD1oIMKSw6Nt9rqsuWQ). Step (2) is required to download and reformat the fMRI data from the original MATLAB file to readable python pickle files. Step (2) also generates contralateral coordinates that is only required for post-analysis if desired to plot anything in contralateral or to calculate gradient indices on the contralateral side.

Note that only subjects 1, 2, 4, 5, 7, 8, 9, 10, and 11 are used. Once embeddings and the formatted fMRI brain data are generated, upload all contents to the Odyssey cluster in order to run experiments. Both the embeddings and brain data are quite large, so it is recommended to store these within your lab's scratch or storage directory and not in your personal directory.

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

Create the sentence representations with different aggregations for the OpenNMT-py layers by running the following code:
```
python create_sentence_representation.py -word_vocab EXAMPLE.vocab.pt -model EXAMPLE.pred.pt -num_layers 4
```

## Generating Random Embeddings (not used in analyses)
If desired to generate random embeddings uniformly distributed within a range, make sure you are within the `opennmt-inspection` directory and run
```
python random_embeddings.py
```

# Cleaning the fMRI data
We need to download the brain fMRI scans (in this case, those captured when reading ```examplesGLM.txt```). The fMRI scans are found [here](https://drive.google.com/drive/folders/1dfwmC6F8FuXlz_3fu2Q1SiSsZR_BY8RP) (you can use [this link](https://github.com/circulosmeos/gdown.pl) to download from drive via curl, wget, etc. ) *Note in the codebase we only regress to subject 1's embeddings because of computational tractability, but this is easily amended* (in ```odyssey_decoding.py``` and ```make_scripts.py```)

Now that we have the embeddings, we still need to convert the subjects' fMRI data (in *.mat* format) into a more readable *.p* format; run
```
python format_for_subject.py --subject_number [X1 X2 X3]
```
where ```X``` is the number of the subject whose *.mat* file you intend to process; note you can process one or more subjects at a time by listing multiple subject numbers (default is just the first subject (subject 1)). Run
```
python format_for_subject.py --help
```

Relevant `.p` files will be generated within the `examplesGLM/` folder for its respective subjects. Original vollangloc and volaal and its contralateral counterparts are saved as `.mat` files upstream one directory for ease of plotting directly in MATLAB.

Note that the `modified_activations.p` file generated for each subject is relatively large. After running this step, the full MATLAB file is no longer required, and I would recommend deleting the `.mat` file or moving the files to a hard drive and/or only referencing the Google Drive when needing the entire `.mat` file.

# Odyssey Cluster

Make sure in the Odyssey Cluster you had downloaded Anaconda for `module load Anaconda3/5.0.1-fasrc02`. Create a virtual environment called `virtualenv` after loading this module and `pip install` relevant packages before running any experiments.

Rsync any files in the `opennmt-inspection` before running experiments. Post-experiments often require downloaded relevant files back to your local workstation for plotting and further analysis.

Examples to upload code into the Odyssey cluster:
```
rsync -avz --update opennmt-inspection/ cjou@odyssey.rc.fas.harvard.edu:~/projects/opennmt-inspection/
```

Examples to download code from cluster to local:
```
rsync -avz cjou@odyssey.rc.fas.harvard.edu:~/projects/mat/ mat/
```

Ensure that the Odyssey cluster directories are set up in a similar manner to the local directory. Scripts generated from code should maintain their own folder and relation to the github repository. Below is the project organization on my personal Odyssey cluster directory.

	├── projects     								 
	│   ├── opennmt-inspection       				<- 
	│   ├── examplesGLM								<- 
	│   ├── mfit									<- 
	│   ├── VBA-toolbox								<- 
	│   └── ...   									<- 
	├── nested_cv          							<- 
	├── decoding_scripts							<- 
	├── logs              							<- 
	└── ...     

Embeddings for the models and fMRI data is also saved within the lab's directory for storage space.

Experiments
------------
1. Setting up the Experiment scripts 
2. Running the Experiment
3. Getting the Experimental results

# Setting up the Experiment scripts
For both experiments, update the relevant file paths for the fMRI data and the save path in `nested_decoding.py` under the argument parser. Update the same respective paths in `convert_np_to_matlab.py`.

## Decoding

Make the nested scripts according to which model. Example as follows:
```
python make_nested_scripts.py -num_layers 12 -bert -email ckjou@college.harvard.edu -file_path projects/ -embedding_path /n/shieber_lab/Lab/users/cjou/embeddings/
```

`make_nested_scripts.py` runs `nested_decoding.py` which computes the nested cross-validation. This updated script version assumes model-to-brain directionality of decoding. Older versions of code that can be referenced are `odyssey_decoding.py` (which does not include nested cross-validation), `no_spotlight_decoding.py` (which does not compute spotlights but by voxel-to-activation). 

Likelihood, ranks, and RMSEs are calculated and saved for model-to-brain, and likelhood and RMSE is saved for brain-to-model. The default saving more is for likelihood and RMSE only; to compute ranking, a flag must be added. The mixed effects analysis, dependent on data, does not converge; do not add this flag.

## Representational Similarity Analysis (RSA)

Computation for RSA is the same as the above part for `make_nested_scripts.py` with an added `-rsa` flag. Make relevent nested scripts according to model as follows:

```
python make_nested_scripts.py -num_layers 12 -bert -email ckjou@college.harvard.edu -file_path projects/ -embedding_path /n/shieber_lab/Lab/users/cjou/embeddings/ -rsa
```
# Running the Experiment

Once all the scripts are generated for the model type and experiment desired, rsync the scripts to the Odyssey cluster. Then from the `nested_cv/` directory, the experiments should be organized by folder. Both locally and on the cluster, the folder organization looks like the following:

	├── nested_cv     								 
	│   ├── nested_cv_bert_subj1 
	│	│   ├── nested_cv_bert_subj1.sh       								
	│	│   ├── nested_cv_bert_subj1_layer1.sh       
	│	│   ├── nested_cv_bert_subj1_layer2.sh 
	│	│   ├── nested_cv_bert_subj1_layer3.sh 	   
	│	│   └── ...  
	│   ├── nested_cv_bert_subj2	
	│	│   ├── nested_cv_bert_subj2.sh       								
	│	│   ├── nested_cv_bert_subj2_layer1.sh       
	│	│   ├── nested_cv_bert_subj2_layer2.sh 
	│	│   ├── nested_cv_bert_subj2_layer3.sh 	   
	│	│   └── ...  	
	│   ├── nested_cv_bert_subj3
	│	│   ├── nested_cv_bert_subj3.sh       								
	│	│   ├── nested_cv_bert_subj3_layer1.sh       
	│	│   ├── nested_cv_bert_subj3_layer2.sh 
	│	│   ├── nested_cv_bert_subj3_layer3.sh 	   
	│	│   └── ...  						
	│   └── ... 
	├── nested_cv_rsa
	└── ...     

Navigate to the proper folder and run the master script (which should have the same name as the folder).

```
cd nested_cv/nested_cv_bert_subj1/
sbatch nested_cv_bert_subj1.sh
```
That should launch the experiment subscripts within that corresponding folder. After running the master scripts for each model experiment and ensuring that all parts of the experiment are successfully completed read `Getting the Experimental Results` below the `Decoding` and `RSA` sections. This concatenates the experimental results from the batches since a pipeline was not completely built to retrigger experiments that failed. Original work in progress in `pipeline.py`.

# Getting the Experimental results

Once the all parts of the experiments are completed, the results need to be concatenated (for LLH, RMSEs, ranking, etc). Check to make sure the file paths in the argparser have been updated and run the code corresponding to the experiment.

```
python convert_np_to_matlab.py -bert -num_layers 12 -model_to_brain
python convert_np_to_matlab.py -bert -num_layers 12 -rsa
```

This will generate the proper MATLAB file of the experimental results in 3D brain space in the folder `../mat/`.

Analyses Overview
------------
1. Bayesian Model Comparison
2. RSA by region
3. RSA by spotlight (slope/argmax)

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
Run `calculate_slope_maps.py` with flags to specify if for language ROIs or all atlas AAL (-aal) and if for slope or argmax analysis
```
python calculate_slope_maps.py -null -slope -local
python calculate_slope_maps.py -null -argmax -aal
```
The model defaulted for the RSA analysis is currently BERT. The `-anova` flag currently does not work.

Histograms for thresholded significant values at 0.01, 0.05, and 0.1 should be generated in the `../visualizations/` folder as well as corresponding bar graphs for whichever analysis specified.

# RSA by spotlight (slope/argmax)
Run `calculate_slope_maps.py` with flags to specify if for language ROIs or all atlas AAL (-aal) and if for slope or argmax analysis
```
python calculate_slope_maps.py -slope -local
python calculate_slope_maps.py -argmax -aal
```
The usage of the analysis is similar to the part above of `RSA by region` except we do not include the `-null` flag.

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
