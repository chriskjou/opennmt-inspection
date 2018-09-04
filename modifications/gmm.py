'''
Gaussian Mixture Model code.
============================

This code is meant to be imported by another Python script to be run.

Here is an example of usage. This will test all neurons in 'en-es-1.desc'
for f1 selectivity for words that end in 's'. It can also be found in gmm_example.py

```
import gmm
import json

gmm.initialize(['en-es-1.desc.t7'], 'en.tok')

def my_tagger(sentence):
    return [(word[-1] == 's') for word in sentence]

f1_scores = gmm.run_gmm(my_tagger, (True, False),
    desc='ends in s', scoring_function=gmm.make_f1_scorer(0))

json.dump(f1_scores, open('output.json', 'w'))
```

'''
import torch
from itertools import product as p
from tqdm import tqdm
import json
import os

def load_dump(fname, cpu_only = False):
    print('Attempting to load dump', fname)

    # Load as 4000x(sentence_length)x(layers)x[500] array
    # (where [] indicates that this is a Tensor)
    if cpu_only:
        raw_dump = [[[layer.cpu() for layer in token] for token in sentence] for sentence in tqdm(torch.load(fname), desc="Move to cpu")]
    else:
        raw_dump = torch.load(fname)

    shape = (len(raw_dump[0][0]), raw_dump[0][0][0].shape[0])

    # Concatenate into 4000x[(sentence_length)x(total_neurons)]
    layers_concatenated = [
        torch.stack([
            torch.cat(token_layers)
            for token_layers in sentence
        ]) for sentence in tqdm(raw_dump, desc="Concatenate")]

    # Concatenate into [(total_tokens)x(total_neurons)]
    sample = torch.cat(layers_concatenated)

    # Normalize description
    mean = sample.mean(0)
    stdev = (sample - mean).pow(2).mean(0).sqrt()

    # In-place normalize
    network = [(x - mean / stdev) for x in tqdm(layers_concatenated, desc="Whiten")]

    return shape, network

# ACCURACY SCORE
# ==============
# A scoring function for use in run_gmm
def accuracy_score(indices, tag_tensor):
    return indices.eq(tag_tensor.unsqueeze(1).expand_as(indices)).float().mean(0)

# MAKE_F1_SCORER
# ==============
# A factory function that produces scoring functions for use in run_gmm
# Usage: run_gmm(my_tagger, my_tag_list, scoring_function = make_f1_scorer(0))
# Where 0 is the relevant class.
def make_f1_scorer(index):
    epsilon = 1e-7

    def f1_score(indices, tag_tensor):
        positives = indices.eq(index).float()
        retrieved = tag_tensor.eq(index).unsqueeze(1).expand_as(indices).float()

        precision = (positives * retrieved).sum(0) / (epsilon + retrieved.sum(0))
        recall = (positives * retrieved).sum(0) / (epsilon + positives.sum(0))

        return 2 * (precision * recall) / (precision + recall + epsilon)
    return f1_score

# RUN_SINGLE_GMM
# ==============
# Same as run_gmm, but for running without "initialize" when you don't want to
# run many tags on many networks at once.
def gmm(
        source=None,
        dump=None,
        manual_tag=None,
        tags=None,
        desc='match',
        use_gpu=False,
        simulate_balanced=False,
        scoring_function=accuracy_score):

    # Tag to index:
    tag2idx = {tag: i for i, tag in enumerate(tags)}

    concatenated_tags = []

    used_tags = set()

    # Load dump
    (layers, n_neurons), network = load_dump(dump, cpu_only=not use_gpu)

    def get_neuron_coordinate(i):
        return (i // n_neurons, i % n_neurons)

    # Sort into buckets
    for i, line in tqdm(enumerate(source), total=len(source), desc='Tag %s' % desc):
        # Assert that lengths agree
        if network[i].shape[0] != len(line):
            raise Exception(
                '''Dump line length does not agree with
                source line length (line %d)''' % i)
        line_tags = manual_tag(line)
        used_tags.update(line_tags)
        concatenated_tags.extend([tag2idx[tag] for tag in line_tags])

    tag_tensor = torch.Tensor(concatenated_tags).long()
    if use_gpu:
        tag_tensor = tag_tensor.cuda()

    # tokens x dim_size
    data = torch.cat(network).float()
    if use_gpu:
        data = data.cuda()
    tokens, dim_size = data.size()

    # Get necessary data for mixed Gaussian model
    mean_tensor = torch.stack([
        data.index_select(0, tag_tensor.eq(i).nonzero().squeeze()).mean(0)
        for i in range(len(tags))
    ])

    stdev_tensor = torch.stack([
        data.index_select(0, tag_tensor.eq(i).nonzero().squeeze()).std(0)
        for i in range(len(tags))
    ])

    count_tensor = torch.Tensor([tag_tensor.eq(i).float().mean() for i in range(len(tags))])
    if use_gpu:
        count_tensor = count_tensor.cuda()

    count_tensor = torch.log(count_tensor)

    # Do predictions from mixed Gaussian model
    likelihoods = data.unsqueeze(0).expand(len(tags), tokens, dim_size)

    mean_tensor = mean_tensor.unsqueeze(1)
    stdev_tensor = stdev_tensor.unsqueeze(1)
    count_tensor = count_tensor.unsqueeze(1).unsqueeze(1)

    # Proper Gaussian likelihood is:
    # e^( -1/2 ((x - mu) / sigma)^2 ) * 1/(sqrt(2 pi) * sigma)
    # => e^( 1/2 ((x - mu) / sigma)^2 ) * 1 / sigma * 1/sqrt(2pi)
    # Hence log likelihood is:
    # = -1/2 ((x - mu) / sigma))^2 - log(sigma) + C
    # where C is a normalizing term that can be discarded.
    #
    # Applying Bayes' theorem we want to add count_tensor to this to get
    # posterior predictions.
    likelihoods = (-(
        (likelihoods - mean_tensor) / stdev_tensor
    ) ** 2) / 2 - torch.log(stdev_tensor)

    # simulate_balanced can be turned on for corpora with very
    # unbalanced class counts, where we would like the predictor
    # to pretend the classes are balanced.
    if not simulate_balanced:
        likelihoods += count_tensor

    # Indices here should be tokens x dim_size
    maxs, indices = torch.max(likelihoods, dim = 0)

    # Accuracies
    accuracies = scoring_function(indices, tag_tensor)

    scores, neurons = torch.sort(accuracies, descending = True)

    scores = scores.cpu().numpy().tolist()
    neurons = neurons.cpu().numpy().tolist()
    neurons_f = [get_neuron_coordinate(i) for i in neurons]

    return list(zip(neurons_f, neurons, scores)), (mean_tensor.squeeze(), stdev_tensor.squeeze())
