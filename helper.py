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
		plabel = "permutation"
	else:
		plabel = ""

	if args.permutation_region:
		prlabel = "permutation_region"
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