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

print(sys.argv)

def log(x):
    sys.stdout.flush()
    tqdm.write(x)
    sys.stdout.flush()

def single_target_modification(
        model = '',
        source = [],
        target = [],
        gold = [],

        num_neurons = 1,

        alignment_training_source = [],
        alignment_training_target = [],

        searching_corpus_source = [],
        searching_corpus_target = [],
        searching_dump = '',

        property_taggers = [],
        target_property = -1,

        magnitude_cap = 1,

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

    ranking, params = search(
        searching_corpus_source,
        tags,
        searching_dump,
        property_projection = lambda s, t: any(y[target_property] for y in t)
    )

    neurons, indices, _ = zip(*ranking[:num_neurons])

    # Collect together Gaussian parameters
    means, stdevs = params
    positive_means = [means[0, index] for index in indices]
    negative_means = [means[1, index] for index in indices]
    positive_stdevs = [stdevs[0, index] for index in indices]
    negative_stdevs = [stdevs[1, index] for index in indices]

    # Compute optimal sample poitns
    optimal_points = [
        (mu_1 * sigma_2 ** 2 - mu_2 * sigma_1 ** 2) /
        (sigma_2 ** 2 - sigma_1 ** 2)
        for mu_1, mu_2, sigma_1, sigma_2 in zip(
            positive_means,
            negative_means,
            positive_stdevs,
            negative_stdevs
        )
    ]

    # Clamp optimal points to within magnitude cap,
    # and round for caching purposes
    sample_points = [
        min(max(x,
            mu_1 - sigma_1 * magnitude_cap),
            mu_1 + sigma_1 * magnitude_cap)
        for x, mu_1, sigma_1 in zip(
            optimal_points,
            positive_means,
            positive_stdevs
        )
    ]

    # Make serializable and round for caching purposes
    sample_points = [
        round(x.numpy().tolist(),
            -int(floor(log10(abs(x.numpy().tolist())))) + 2)
        for x in sample_points
    ]

    log('Using neurons %s' % ', '.join('(%d, %d)' % neuron for neuron in neurons))

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
            (source_line, domain_locus[i]) for i, x in enumerate(source_line)
            if domain_locus[i] is not None
        ])
        example_gold.extend([
            gold_line for i, x in enumerate(source_line)
            if domain_locus[i] is not None
        ])
    example_sources, _ = zip(*example_corpus)

    log('Found %d examples.' % len(example_corpus))

    # Get the usual activation of our selected neuron under
    # the target property

    log('Using sample values %s' % (', '.join('%f' % x for x in sample_points),))

    # Modify these examples to match the target property
    log('MODIFYING')
    log('=========')

    modified = modify(
        corpus = example_corpus,
        neurons = neurons,
        values = sample_points,
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
        len(example_sources)
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
    for (example, (target, _)), tags in zip(example_corpus, modified_tags):
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

    def tag_singular(tokens):
        return [p == 'NOUN' and
            'Number' in m and m['Number'] == 'Sing'
            for p, m in tag_es(tokens)]

    def tag_plural(tokens):
        return [p == 'NOUN' and
            'Number' in m and m['Number'] == 'Plur'
            for p, m in tag_es(tokens)]

    # Only try to change to present those verbs
    # that are currently aligned to past and not
    # to present.
    def tag_past_verbs(source_tokens, aligned_results):
        doc = Doc(vocab=en.vocab, words=source_tokens)
        en.tagger(doc)

        return [(i,(i,)) if t.pos_ == 'VERB' and
            any(m[0] for m in r) and not
            any(m[1] for m in r) else None for i, (t, (_, r))
            in enumerate(zip(doc, aligned_results))]

    # Only try to change to plural those nouns
    # currently aligned to singular and not plural
    def tag_relevant_nouns(source_tokens, aligned_results):
        doc = Doc(vocab=en.vocab, words=source_tokens)
        en.tagger(doc)
        en.parser(doc)

        return [
            (j, tuple(i for i, d in enumerate(doc)
                if d == t or d.head == t))
            if t.pos_ == 'NOUN' and
            any(m[0] for m in r) and not
            any(m[1] for m in r) else None for j, (t, (_, r))
            in enumerate(zip(doc, aligned_results))]

    def load_tokenized(fname):
        result = []
        with open(fname) as f:
            for line in f:
                result.append(line.strip().split(' '))
        return result

    # Change-to-present BRNN tense experiment.
    if sys.argv[1] == 'tense':
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

            domain = tag_past_verbs
        )

    # Change-to-plural RNN number experiment
    elif sys.argv[1] == 'number':
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

            property_taggers = [tag_plural, tag_singular],
            target_property = 1,
            magnitude_cap = 20,
            num_neurons = 5,

            domain = tag_relevant_nouns
        )
