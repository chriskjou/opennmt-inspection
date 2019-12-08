import torch
import numpy as np

 # def extract_features_aligned_to_words(model, sentence: str, return_all_hiddens: bool = False) -> torch.Tensor:
	# 	"""Extract RoBERTa features, aligned to spaCy's word-level tokenizer."""
	# 	from fairseq.models.roberta import alignment_utils
	# 	from spacy.tokens import Doc

	# 	nlp = alignment_utils.spacy_nlp()
	# 	tokenizer = alignment_utils.spacy_tokenizer()

	# 	# tokenize both with GPT-2 BPE and spaCy
	# 	bpe_toks = model.encode(sentence)
	# 	spacy_toks = tokenizer(sentence)
	# 	spacy_toks_ws = [t.text_with_ws for t in tokenizer(sentence)]
	# 	alignment = alignment_utils.align_bpe_to_words(self, bpe_toks, spacy_toks_ws)

	# 	# extract features and align them
	# 	features = self.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
	# 	features = features.squeeze(0)
	# 	aligned_feats = alignment_utils.align_features_to_words(self, features, alignment)

	# 	# wrap in spaCy Doc
	# 	doc = Doc(
	# 		nlp.vocab,
	# 		words=['<s>'] + [x.text for x in spacy_toks] + ['</s>'],
	# 		spaces=[True] + [x.endswith(' ') for x in spacy_toks_ws[:-1]] + [True, False],
	# 	)
	# 	assert len(doc) == aligned_feats.size(0)
	# 	doc.user_token_hooks['vector'] = lambda token: aligned_feats[token.i]
	# 	return doc
	
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large') # force_reload=True)
roberta.eval() 

tokens = roberta.encode('how does this work')
features = roberta.extract_features(tokens, return_all_hiddens=True)
print(type(features))
features = features[0].squeeze(0)
print(len(features))

doc = roberta.extract_features_aligned_to_words('the kid ate the bob', return_all_hiddens=True)
# assert last_layer_features.size() == torch.Size([1, 5, 1024])
print(doc.size())

# tokens = roberta.encode('how does this work')
# last_layer_features = roberta.extract_features_aligned_to_words('how does this work') #, return_all_hiddens=True)
# print(len(last_layer_features))

print("done.")