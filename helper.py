import numpy as np
from tqdm import tqdm
import scipy.io

def validate_arguments(args):
	if args.language not in languages:
		print("invalid language")
		exit()
	if args.num_layers not in num_layers:
		print("invalid num_layer")
		exit()
	if args.model_type not in model_type:
		print("invalid model_type")
		exit()
	if args.agg_type not in agg_type:
		print("invalid agg_type")
		exit()
	if args.subj_num not in subj_num:
		print("invalid subj_num")
		exit()
	if args.which_layer not in list(range(args.num_layers)):
		print("invalid which_layer: which_layer must be between 1 and args.num_layers, inclusive")
		exit()
	if args.brain_to_model and args.model_to_brain:
		print("select only one flag for brain_to_model or model_to_brain")
		exit()
	if not args.brain_to_model and not args.model_to_brain:
		print("select at least one flag for brain_to_model or model_to_brain")
		exit()
	if (args.brain_to_model or args.model_to_brain) and not args.decoding:
		print("select decoding option for model to brain or brain to model")
		exit()
	if not args.decoding and not args.fdr and not args.llh and not args.rank:
		print("select at least one evaluation metric")
		exit()
	return

def generate_labels(args):
	if args.brain_to_model:
		direction = "brain2model_"
	else:
		direction = "model2brain_"

	if args.cross_validation:
		validate = "cv_"
	else:
		validate = "nocv_"
	if args.random:
		rlabel = "random"
	else:
		rlabel = ""
		
	if args.rand_embed:
		elabel = "rand_embed"
	else:
		elabel = ""

	if args.glove:
		glabel = "glove"
	else:
		glabel = ""

	if args.word2vec:
		w2vlabel = "word2vec"
	else:
		w2vlabel = ""

	if args.bert:
		bertlabel = "bert"
	else:
		bertlabel = ""

	if args.permutation:
		plabel = "permutation_"
	else:
		plabel = ""

	if args.permutation_region:
		prlabel = "permutation_region_"
	else:
		prlabel = ""

	return direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel

def generate_options(args):

	get_residuals_and_make_scripts = " --num_batches" + str(args.num_batches)
	if not args.rand_embed and not args.bert and not args.word2vec and not args.glove:
		options = (
			"--language " + str(args.language) +
			" --num_layers " + str(args.num_layers) +
			" --model_type " + str(args.type) +
			" --which_layer " + str(args.which_layer) +
			" --agg_type " + str(args.agg_type) +
			" --subject_number " + str(args.subj_num)
		)
	if args.rand_embed:
		options += " --rand_embed"
	if args.glove:
		options += " --glove"
	if args.word2vec:
		options += " --word2vec"
	if args.bert:
		options += " --bert"

	if args.cross_validation:
		options += " --cross_validation"
	if args.brain_to_model:
		options += " --brain_to_model"
	if args.model_to_brain:
		options += " --model_to_brain"
	if args.random:
		options += " --random"
	if args.permutation:
		options += " --permutation"
	if args.permutation_region:
		options += " --permutation_region"

	return get_residuals_and_make_scripts, options

# transform coordinates for SPM plotting
def transform_coordinates(rmses, volmask, save_path="", metric="", pvals=[]):
	i,j,k = volmask.shape
	nonzero_pts = np.transpose(np.nonzero(volmask))
	metric_vals = np.zeros((i,j,k))
	for pt in tqdm(range(len(nonzero_pts))):
		x,y,z = nonzero_pts[pt]
		metric_vals[int(x)][int(y)][int(z)] = rmses[pt]
	scipy.io.savemat(save_path + "-3dtransform-" + str(metric) + ".mat", dict(metric = metric_vals))
	if metric == "rmses":
		scipy.io.savemat(save_path + "-3dtransform-" + str(metric) + "-log.mat", dict(metric = np.log(metric_vals)))
	if metric == "fdr":
		scipy.io.savemat(save_path + "-3dtransform-" + str(metric) + "-pvals.mat", dict(metric = pvals))
	# pickle.dump( metrics, open(save_path + "-3dtransform-" + str(metric) + ".p", "wb" ) )
	# pickle.dump( np.log(metrics), open(save_path + "-3dtransform-log-" + str(metric) + ".p", "wb" ) )
	return metric_vals

# clean ROI labels for plotting and ranking
def clean_roi(roi_vals, roi_labels):
	roi_vals = roi_vals.reshape((len(roi_vals), ))
	final_roi_labels = []
	for val_index in roi_vals:
		if val_index == 0:
			final_roi_labels.append("other")
		else:
			final_roi_labels.append(roi_labels[val_index-1][0][0])
	return final_roi_labels

# clean atlas labels for plotting and ranking
def clean_atlas(atlas_vals, atlas_labels):
	at_vals = atlas_vals.reshape((len(atlas_vals), ))
	at_labels = []
	for val_index in at_vals:
		at_labels.append(atlas_labels[val_index-1][0][0])
	return at_labels

# create bash scripts for RANK and FDR
def create_bash_script(args, fname, file_to_run, memory, time_limit, batch=-1, total_batches=-1, cpu=1):
	direction, validate, rlabel, elabel, glabel, w2vlabel, bertlabel, plabel, prlabel = generate_labels(args)

	if args.brain_to_model:
		dflag = " --brain_to_model "
	elif args.model_to_brain:
		dflag = " --model_to_brain "
	else:
		dflag = ""

	if args.cross_validation:
		cvflag = " --cross_validation "
	else:
		cvflag = ""

	sub_flag = " --subject_number {} ".format(args.subject_number)
	pflag = "" if (plabel == "") else "--" + str(plabel)
	prflag = "" if (prlabel == "") else "--" + str(prlabel)
	rflag = "" if (rlabel == "") else "--" + str(rlabel)
	gflag = "" if (glabel == "") else "--" + str(glabel)
	w2vflag = "" if (w2vlabel == "") else "--" + str(w2vlabel)
	bertflag = "" if (bertlabel == "") else "--" + str(bertlabel)
	eflag = "" if (elabel == "") else "--" + str(elabel)
	partition = "serial_requeue" if not args.gpu else "seas_dgx1"
	num_cpu = "" if (cpu == 1) else "#SBATCH -n {} ".format(cpu)
	batch_num = "" if (batch == -1) else " --batch_num {} ".format(batch)
	total_batch_num = "" if (total_batches == -1) else " --total_batches {} ".format(total_batches)
	
	with open(fname, 'w') as rsh:
		rsh.write('''\
#!/bin/bash
#SBATCH -J {0}  								# Job name
#SBATCH -p {1} 									# partition (queue)
#SBATCH --mem {2} 								# memory pool for all cores
#SBATCH -t {3} 									# time (D-HH:MM)
{4}
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../logs/outpt_{0}.txt 			# File that STDOUT writes to
#SBATCH -e ../../logs/err_{0}.txt				# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python {5} \
--embedding_layer /n/shieber_lab/Lab/users/cjou/embeddings/parallel/{6}/{7}layer-{8}/{9}/parallel-english-to-{6}-model-{7}layer-{8}-pred-layer{10}-{9}.mat \
{11} {12} {13} {14} {15} {16} {17} {18} {19} {20} {21} {22} {23}
'''.format(
		job_id, 
		partition,
		memory, 
		time_limit,
		num_cpu,
		file_to_run,
		args.language, 
		args.num_layers, 
		args.model_type, 
		args.agg_type, 
		args.which_layer, 
		dflag,
		cflag,
		batch_num, 
		total_batch_num,
		sub_flag,
		rflag,
		eflag,
		gflag,
		w2vflag,
		bertflag,
		pflag,
		prflag
	)
)