# Project Organization
------------
Folders will be generated one directory back from this current github repository; to keep all data, dependent program, and code centralized, it would be best to put this repo within its own folder. For example, part of my current setup is as shown below in a folder called `research`. When running on the Odyssey cluster, make sure the paths are still relative to each other in the same way or directly change the paths in the code.

    └─ research
        ├── opennmt-inspection                          <- current github repo
        │   ├── scripts                                 <- added temporary scripts for testing
        │   ├── notebooks                               <- added jupyter notebooks
        │   └── ...                                     <- all other folders were from original branch
        ├── bspmview                                    <- dependency for MATLAB plotting
        ├── examplesGLM                                 <- fMRI data downloaded from Google Drive
        │   ├── subj1                                   <- subfolder for subj1
        │   ├── subj2       
        │   ├── subj3    
        │   └── ...        
        ├── embeddings                                  <- folder of embeddings generated from code
        │   ├── bert                                    <- subfolder of generated embeddings
        │   ├── glove       
        │   ├── gpt2    
        │   └── ...        
        ├── ccnl-fmri                                   <- dependency for MATLAB plotting
        ├── Language-fMRI                               <- dependency for MATLAB plotting
        ├── spm12                                       <- dependency for MATLAB plotting
        ├── VBA-toolbox                                 <- dependency for BMS library
        ├── visualizations                              <- folder of graphs generated from code
        ├── decoding_scripts                            <- folder of generated scripts for Odyssey from code
        │   ├── model1       
        │   ├── model2 
        │   └── ...   
        ├── mat                                         <- downloaded folder from Odyssey
        ├── GoogleNews-vectors-negative300.bin.gz       <- downloaded word2vec dependency
        ├── glove.6B                                    <- downloaded glove dependency
        └── ...           

Important added files are briefly described here with a more thorough description provided on file or in workflow.

    └─ opennmt-inpsection
        ├── calculate_slope_maps.py                     <- calculate argmax/slope across layers for a voxel
        ├── find_best_likelihood.py                     <- prepare LLH for mfit or BMS model comparison in VBA
        ├── flair_embeddings.py                         <- embeddings from Huggingface NLP models
        ├── format_for_subject.py                       <- formatting fMRI data from MATLAB to python
        ├── get_pretrained_embeddings.py                <- embeddings from GloVe and word2vec
        ├── helper.py                                   <- important helper functions
        ├── make_nested_scripts.py                      <- script generation for decoding/RSA with spotlights with nested cross-validation
        ├── make_neurosynth_rsa_scripts.py              <- script generation for RSA by language region
        ├── make_scripts.py                             <- make scripts for decoding or RSA experiments with spotlight (original)
        ├── metric_across_layers.py                     <- 2D plot of a metric across layers in a NLP model
        ├── nested_convert_np_to_matlab.py              <- convert pickle files from batches into MATLAB 3D space
        ├── nested_cv_significance.py                   <- create count/average graphs for BMS model family comparison (VBA-toolbox)
        ├── nested_decoding.py                          <- updated decoding/RSA experiment from odyssey_decoding with nested cross-validation
        ├── nested_vba_toolbox.py                       <- generate plots from PXP/BOR values in VBA BMS family
        ├── neurosynth_rsa.py                           <- RSA experiment by language region
        ├── plot_initial_activations.py                 <- plot graphs for initial fMRI activations
        ├── plot_initial_embeddings.py                  <- plot graphs for initial embeddings (pretrained, NLP, opennmt)
        ├── plot_residuals_locations.py                 <- [deprecated] original 2D plot of metric across models
        ├── significance_threshold.py                   <- [deprecated] original analysis of GLM across scanner runs
        ├── significant_llh.py                          <- [deprecated] original concatenate and plot mfit PXP/BOR and prepare for transform into 3D brain space
        └── ...    
# Dependencies
Libaries that are used and required in all files that were added to the original opennmt-py fork are in `brain_nn_requirements.txt`. Original requirements maintained by the original opennmt-py fork are in `requirements.txt`.

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

## Generating Rando/Permutation Embeddings (not used in analyses)
If desired to generate random embeddings uniformly distributed within a range, make sure you are within the `opennmt-inspection` directory and run
```
python random_embeddings.py
```

Permutation embeddings also generated from `permutation.py`.

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
    │   ├── opennmt-inspection                      <- current repo
    │   ├── examplesGLM                             <- fMRI data (can also be referenced from cluster storage instead)
    │   ├── mfit                                    <- github from mfit (see under analysis)
    │   ├── VBA-toolbox                             <- github dependency for BMS model family analysis
    │   └── ...                                     
    ├── nested_cv                                   <- example location for scripts for nested cross-validation decoding/RSA
    ├── decoding_scripts                            <- example location for scripts for non-nested cross-validation decoding/RSA
    ├── logs                                        <- empty folder for logs generated from Odyssey batch jobs
    └── ...     

Embeddings for the models and fMRI data is also saved within the lab's directory for storage space.

Experiments
------------
1. Setting up the Experiment scripts 
2. Running the Experiment
3. Getting the Experimental results

# Setting up the Experiment scripts
For both experiments, update the relevant file paths for the fMRI data and the save path in `nested_decoding.py` under the argument parser. Update the same respective paths in `nested_convert_np_to_matlab.py`.

## Decoding

Make the nested scripts according to which model. Example as follows:
```
python make_nested_scripts.py -num_layers 12 -bert -email ckjou@college.harvard.edu -file_path projects/ -embedding_path /n/shieber_lab/Lab/users/cjou/embeddings/
```

`make_nested_scripts.py` runs `nested_decoding.py` which computes the nested cross-validation. This updated script version assumes model-to-brain directionality of decoding. Older versions of code that can be referenced are `odyssey_decoding.py` (which does not include nested cross-validation), `no_spotlight_decoding.py` (which does not compute spotlights but by voxel-to-activation). 

Likelihood, ranks, and RMSEs are calculated and saved for model-to-brain, and likelhood and RMSE is saved for brain-to-model. The default saving more is for likelihood and RMSE only; to compute ranking, a flag must be added. The mixed effects analysis, dependent on data, does not converge; do not add this flag.

## Representational Similarity Analysis (RSA)

### Spotlights
Computation for RSA is the same as the above part for `make_nested_scripts.py` with an added `-rsa` flag. Make relevent nested scripts according to model as follows:

```
python make_nested_scripts.py -num_layers 12 -bert -email ckjou@college.harvard.edu -file_path projects/ -embedding_path /n/shieber_lab/Lab/users/cjou/embeddings/ -rsa
```
### By Region

Specifically for calculating RSA by region (by Federenko's labels), generate scripts for RSA for BERT as
```
python make_neurosynth_rsa_scripts.py -subject_number 1
```

Scripts will be generated in the same way as with RSA spotlights but under the folder `neurosynth_rsa/`. Run once as dictated by the next part and then again by modifying the flag in `neurosynth_rsa.py` for `null` to be `True`. The first run without the flag will calculate the RSA correlations per region, and then with the flag, will calculate the standard deviation, mean, and pval for 100 trials randomly permutated.

# Running the Experiment

Once all the scripts are generated for the model type and experiment desired, rsync the scripts to the Odyssey cluster. Then from the `nested_cv/` directory, the experiments should be organized by folder. Both locally and on the cluster, the folder organization looks similar to the following:

    ├── nested_cv                                    
    │   ├── nested_cv_bert_subj1 
    │   │   ├── nested_cv_bert_subj1.sh                                     
    │   │   ├── nested_cv_bert_subj1_layer1.sh       
    │   │   ├── nested_cv_bert_subj1_layer2.sh 
    │   │   ├── nested_cv_bert_subj1_layer3.sh     
    │   │   └── ...  
    │   ├── nested_cv_bert_subj2    
    │   │   ├── nested_cv_bert_subj2.sh                                     
    │   │   ├── nested_cv_bert_subj2_layer1.sh       
    │   │   ├── nested_cv_bert_subj2_layer2.sh 
    │   │   ├── nested_cv_bert_subj2_layer3.sh     
    │   │   └── ...     
    │   ├── nested_cv_bert_subj3
    │   │   ├── nested_cv_bert_subj3.sh                                     
    │   │   ├── nested_cv_bert_subj3_layer1.sh       
    │   │   ├── nested_cv_bert_subj3_layer2.sh 
    │   │   ├── nested_cv_bert_subj3_layer3.sh     
    │   │   └── ...                         
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

Once the all parts of the experiments are completed, the results need to be concatenated (for LLH, RMSEs, ranking, etc). Check to make sure the file paths in the argparser have been updated and run the code corresponding to the experiment. For specifically concatenating the LLH for the decoding of model-to-brain with nested cross-validation, run 

```
python nested_convert_np_to_matlab.py -bert -subject_number 1           # model-to-brain LLH
python nested_convert_np_to_matlab.py -bert -subject_number 1 -rsa      # RSA
```

This will generate the proper MATLAB file of the experimental results in 3D brain space in the folder `../mat/`. Reference code if RSA takes too long or needs to check previous code, files without the `nested` prefix can be used instead. For example, instead of `make_nested_scripts.py`, `make_scripts.py` can be used; instead of `nested_decoding.py`, `odyssey_decoding.py` can be used; and instead of `nested_convert_np_to_matlab.py`, `convert_np_to_matlab.py` can be used. The original files without `nested` include batching instead and for decoding, includes nested cross validation (but is not applicable to RSA).

Analyses Overview
------------
1. Bayesian Model Comparison
2. RSA by region
3. RSA by spotlight (slope/argmax)
4. GLM by scanner runs (older analysis not shown in slides)

# Bayesian Model Comparison (BMS)

## Preparing the data
Ensure that the entire experiment for all subjects and all layers of models have successfully run and are concatenated.

[COMPARE MODELS] To generate a MATLAB file in preperation for VBA-toolbox, make sure the BERT, baseline, and opennmt models have successfully run. Then execute
```
python find_best_likelihood.py -local -save_by_voxel -compare_models -llh
```
This will create the proper MATLAB files in the format of `voxel x model layers x number of subjects` for `VBA_groupBMC` in the `mfit/` directory (which was originally used in a previous analysis) and will be accessed when running the VBA-toolbox scripts.

[BERT ONLY - not used - mfit only] To generate the same MATLAB file but only for values in the BERT model, do the following,
```
python find_best_likelihood.py -local -save_by_voxel -llh
```

[not used - mfit only] After running the entire experiment for all subjects and all layers of a particular model (such as BERT), we can put the LLH values into a MATLAB file with 
```
python find_best_likelihood.py -local -single_subject -across_layer -subject_number 1 -which_layer 1 -bert _num_layers 12 -llh
```

This generates 12 MATLAB files (one for each layer) with boolean values denoting whether that layer was the best fit for that particular voxel for that specific subject.

## Generating scripts
Make sure that [VBA toolbox](https://github.com/MBB-team/VBA-toolbox) is downloaded upstream one directory from `opennmt-inspection` both locally and on the cluster if desired to run there.

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

## Comparing and visualizing model families

To generate a count graph for all significant voxels in each language ROI for best representation of a model (bert vs. baseline vs. opennmt), run
```
python nested_cv_significance.py -local -count
```
After you have run the code at least once, you can run commands with the flag `-use_cache` to avoid computing ANOVA pvalue significances again.

Other uses of plotting include averaging the LLH metric across all significant voxels in each brain atlas AAL and then plotting the metric value for each of the models as
```
python nested_cv_significance.py -local -aal -avg
```

## [Deprecated from] mfit
[mfit](https://github.com/sjgershm/mfit) replaced by VBA-toolbox analysis.
Original files to plot BOR and PXP per subject/layer in 2D and 3D brain space: `significant_llh.py`

# RSA by region 

## Correlation Plots
To plot the RSA correlations of the language ROI or all atlas AAL for BERT, run the following with corrected null distrbution for one subject or across subject as:
```
python null_rsa_distribution.py -subject_number 1
python null_rsa_distribution.py -across_subjects -aal
```

## Slope and Argmax
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

# GLM by scanner runs (older analysis not shown in slides)

Used in `significance_threshold.py` but no conclusive or concrete evidence for any claims. Similar to `mixed_effects.py`.

Extending the Analysis and Visualization
------------
1. Gradient Correlation
2. Plotting in 3D Brain space
3. Plotting initial embeddings and activations
4. Heatmaps across layers
5. Metric across layers (2D)
6. Helpful Functions
7. Older versions of files

# Gradient Correlation
To calculating an anatomical gradient index (1) BMS VBA-toolbox argmax; (2) RSA slope; or (3) RSA argmax, run the following line of code for respective analyses. Add the flag for `-contra` to calculate the gradient index for the same analysis but on the contralateral side of the brain. 
```
python gradient_correlation.py -spearman            # VBA argmax
python gradient_correlation.py -pearson -slope      # RSA slope
python gradient_correlation.py -spearman -argmax    # RSA argmax
```

# Plotting in 3D Brain space
Redirect to the `Language-fMRI` folder. Make sure dependencies of `spm12`, `ccnl-fmri`, and `bspmview` are on the MATLAB path.

Load the corresponding .mat file into MATLAB. Follow the directions under `save_vol.m` to plot values in 3D brain space and generate brain slice/surface rendering photos.

Note: Current plots are not entirely exact due to missing MNI coordinates. Values less than 1 are also often clipped; make sure the proper data values are being plotted and the range on the plot is not cut off.

# Plotting initial embeddings and activations
Graphs are created in `visualizations/`. Ensure that the fMRI data and the embedding paths are updated where noted in each of the files used.

For plotting initial embeddings:
```
python plot_initial_embeddings.py -bert     # plot initial bert embeddings for all aggregations types for specific dimensions 
```

For plotting initial fMRI brain activations:
```
python plot_initial_activations.py -brain_map -subjects 1,2,4,5 -sentences 1,2,3    # saving activations for sentences 1, 2, 3 in common brain space of given subjects
python plot_initial_activations.py -hist -aal -subject_number 1                     # saving histograms for each AAL region for a specific subject
python plot_initial_activations.py -subject_number 1                                # plot graph per language ROI region for specific subject                               
```

# Heatmaps across layers
To generate heatmaps for a single subject across layers for a metric (LLH, ranking, RMSE, etc), run
```
python compare_layers_and_subjects.py -single_subject -across_layers -bert -num_layers 12 -subject_number 1 -rmse
```

To generate heatmaps at a group level across layers for a metric (LLH, ranking, RMSE, etc in the common brain space of the specified subjects, run
```
python compare_layers_and_subjects.py -group_level -across_layer -bert -num_layers 12
```

[deprecated] To generate MATLAB files for 3D brain space to see the heatmap for a single layer of metric values, run
```
python compare_layers_and_subjects.py -group_level -single_layer -which_layer 1
```

# Metric across layers
First, update the default file paths for the fMRI data and the save path or provide as a argument.

To generate a 2D plot of a metric across layers of a particular model, run
```
python metric_across_layers.py -local
```

# Helpful Functions
When converting back a forth between the coordinates in the MATLAB file and in the either same format in Python or unraveled version, check out the helper functions in `helper.py` such as `convert_matlab_to_np` and `convert_np_to_matlab`. An example usage script for transforming coordinates is in `transform_coordinates_for_plotting.py`.

Other helpful functions include finding the common space brain given a set of subjects (`load_common_space`) and cleaning atlas AAL or language ROI labels (`clean_atlas` and `clean_roi`). 

# Older version of files
These files are usually forks off the original file but modified slightly to reduce any code merging problems at the time. In the future, these files can be merged into the one file for optimal pipelining.

`no_spotlight_decoding.py`: for both decoding and RSA experiments except no spotlights and only single voxel activations are used per relationship