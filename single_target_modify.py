from modifications.align import align
from modifications.tag import tag_aligned
from modifications.modify import modify
from modifications.search import search

from math import log10, floor

from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu

import sys

import json

import hashlib

def log(x):
    sys.stdout.flush()
    tqdm.write(x)
    sys.stdout.flush()

def single_target_modification(
        model = '',
        source = [],
        target = [],
        gold = [],

        alignment_training_source = [],
        alignment_training_target = [],

        searching_corpus_source = [],
        searching_corpus_target = [],
        searching_dump = '',

        property_taggers = [],
        target_property = -1,

        magnitude = 1,

        domain = lambda x, t: [True for _ in x]):

    # Align source and target
    log('ALIGNING')
    log('========')

    searching_alignment = align(
        [(searching_corpus_source,
            searching_corpus_target),
         (alignment_training_source,
            alignment_training_target)],
        len(searching_corpus_source)
    )

    active_alignment = align(
        [(source,
            target),
         (alignment_training_source,
            alignment_training_target)],
        len(source)
    )

    # Tag the searching and active corpora
    log('TAGGING')
    log('=======')

    tags = tag_aligned(
        searching_corpus_source,
        searching_corpus_target,
        searching_alignment,
        target_tagger = lambda x: list(zip(*(
            tagger(x) for tagger in property_taggers)))
    )

    active_tags = tag_aligned(
        source,
        target,
        active_alignment,
        target_tagger = lambda x: list(zip(*(
            tagger(x) for tagger in property_taggers)))
    )

    # Search the searching corpus for neurons good at identifying
    # whether something has the given property or not.
    log('SEARCHING')
    log('=========')

    neuron, params = search(
        searching_corpus_source,
        tags,
        searching_dump,
        property_projection = lambda s, t: any(y[target_property] for y in t)
    )

    log('Using neuron (%d, %d).' % neuron)

    # Collect all examples of in-domain tokens we wish to
    # modify
    log('COLLECTING EXAMPLES')
    log('===================')

    example_corpus = []
    example_gold = []

    if gold is None:
        gold = [[] for _ in source]

    for j, (source_line, gold_line) in enumerate(zip(source, gold)):
        domain_locus = domain(source_line, active_tags[j])
        example_corpus.extend([
            (source_line, i) for i, x in enumerate(source_line)
            if domain_locus[i]
        ])
        example_gold.extend([
            gold_line for i, x in enumerate(source_line)
            if domain_locus[i]
        ])
    example_sources, _ = zip(*example_corpus)

    log('Found %d examples.' % len(example_corpus))

    # Get the usual activation of our selected neuron under
    # the target property

    # Default tag order is (True, False) and we want to switch to True,
    # so take that mean.
    mean, std = params[0][0], params[1][0]
    neg_mean, neg_std = params[0][1], params[1][1]

    # Move the neuron somewhat out of its ordinary activation range,
    # away from the negative (i.e. doesn't have desired property) class,
    # by an amount specified by (magnitude)
    sample = (mean + (1 if mean > neg_mean else -1) * std * magnitude).numpy().tolist()

    # Round to 3 significant figures so that small floating point fluctuations
    # don't break the cache
    sample = round(sample, -int(floor(log10(abs(sample)))) + 2)

    log('Using sample value %f' % (sample,))

    # Modify these examples to match the target property
    log('MODIFYING')
    log('=========')

    modified = modify(
        corpus = example_corpus,
        neuron = neuron[1],
        layer = neuron[0],
        value = sample,
        model = model
    )

    # Align and tag the results
    log('ALIGNING MODIFIED RESULT')
    log('========================')
    modified_alignment = align(
        [(example_sources,
            modified),
        (alignment_training_source,
            alignment_training_target)],
        len(source)
    )

    log('TAGGING MODIFIED RESULT')
    log('=======================')

    modified_tags = tag_aligned(
        example_sources,
        modified,
        modified_alignment,
        target_tagger = lambda x: list(zip(*(
            tagger(x) for tagger in property_taggers)))
    )

    # Compute results
    log('RESULTS:')
    log('========')

    result_statistics = {}
    for (example, target), tags in zip(example_corpus, modified_tags):
        classification = frozenset(tuple(x) for x in tags[target][1])
        if classification not in result_statistics:
            result_statistics[classification] = 0
        result_statistics[classification] += 1

    log('Resulting tags:')
    result_pairs = sorted(result_statistics.items(), key = lambda x: -x[1])
    for k, v in result_pairs:
        log('  %r: %d' % (k, v))

    # A "success" is translating to something that is aligned with something
    # *having* the target property and *not having* any other tagged properties. There are two
    # such cases: being aligned to only tokens having only the target property;
    # and being aligned to tokens having the target property and tokens having no properties.

    successful_unique = frozenset([tuple(i == target_property for i in range(len(property_taggers)))])
    successful_trivial = successful_unique.union({tuple(False for _ in property_taggers)})

    successful = sum(result_statistics[x]
            for x in (successful_unique, successful_trivial)
            if x in result_statistics)
    total = sum(result_statistics[x] for x in result_statistics)

    log('Success rate: %d/%d = %f' % (successful, total, successful / total))

    bleu_score = corpus_bleu([[x] for x in example_gold], modified)
    log('Corpus bleu: %f' % bleu_score)

# Change-to-present BRNN tense experiment.
if __name__ == '__main__':

    import spacy
    from spacy.tokens import Doc

    en = spacy.load('en')
    es = spacy.load('es')

    def morph_es(x):
        principal, morph = x.tag_.split('__')
        if morph == '_':
            return principal, {}
        morph = [x.split('=') for x in morph.split('|')]
        return principal, {a: b for a, b in morph}

    def tag_es(tokens):
        doc = Doc(vocab=es.vocab, words=tokens)
        es.tagger(doc)
        return [morph_es(x) for x in doc]

    def tag_past(tokens):
        return [p == 'VERB' and
            'VerbForm' in m and m['VerbForm'] == 'Fin' and
            'Tense' in m and m['Tense'] in ('Past', 'Imp')
            for p, m in tag_es(tokens)]

    def tag_present(tokens):
        return [p == 'VERB' and
            'VerbForm' in m and m['VerbForm'] == 'Fin' and
            'Tense' in m and m['Tense'] in ('Pres',)
            for p, m in tag_es(tokens)]

    # Only try to change to present those verbs
    # that are currently aligned to past and not
    # to present.
    def tag_verbs(source_tokens, aligned_results):
        doc = Doc(vocab=en.vocab, words=source_tokens)
        en.tagger(doc)

        return [t.pos_ == 'VERB' and
            any(m[0] for m in r) and not
            any(m[1] for m in r) for t, (_, r)
            in zip(doc, aligned_results)]

    def load_tokenized(fname):
        result = []
        with open(fname) as f:
            for line in f:
                result.append(line.strip().split(' '))
        return result

    single_target_modification(
        model = 'models/en-es-1.pt',
        source = load_tokenized('un-data/test/en'),
        target = load_tokenized('output/en-es-1.txt'),
        gold = load_tokenized('un-data/test/es'),

        alignment_training_source = load_tokenized('data/UNv1.0.6way.en.tok.2m'),
        alignment_training_target = load_tokenized('data/UNv1.0.6way.es.tok.2m'),

        searching_corpus_source = load_tokenized('un-data/test/en'),
        searching_corpus_target = load_tokenized('output/en-es-1.txt'),
        searching_dump = 'layer-dump/en-es-1-brnn.dump.pt',

        property_taggers = [tag_past, tag_present],
        target_property = 1,
        magnitude = 10,

        domain = tag_verbs
    )

