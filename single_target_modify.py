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

        num_neurons = 1,

        alignment_training_source = [],
        alignment_training_target = [],

        searching_corpus_source = [],
        searching_corpus_target = [],
        searching_dump = '',

        selection_strategy = 'target',

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

    if selection_strategy == 'target':
        ranking, params = search(
            searching_corpus_source,
            tags,
            searching_dump,
            property_projection = lambda s, t: any(y[target_property] for y in t),
            use_gpu = False
        )
    else: # presumably "all"
        ranking, params = search(
            searching_corpus_source,
            tags,
            searching_dump,
            classes = (0, 1, 2),
            simulate_balanced = True,
            property_projection = \
                lambda s, t: (
                    0 if any(y[target_property] for y in t) else
                    1 if any(any(y) for y in t) else
                    2
                )
        )

    neurons, indices, _ = zip(*ranking[:num_neurons])

    # Collect together Gaussian parameters
    means, stdevs = params
    positive_means = [means[0, index] for index in indices]
    negative_means = [means[1, index] for index in indices]
    positive_stdevs = [stdevs[0, index] for index in indices]
    negative_stdevs = [stdevs[1, index] for index in indices]

    # Compute optimal sample poitns
    sample_points = [
        (mu_1 - mu_2) * magnitude_cap + mu_1
        for mu_1, mu_2, sigma_1, sigma_2 in zip(
            positive_means,
            negative_means,
            positive_stdevs,
            negative_stdevs
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
            (source_line, d) for d in domain_locus
        ])
        example_gold.extend([
            gold_line for d in domain_locus
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

    def serialize_tuple(x):
        return ({
            (True, True): '11',
            (False, False): '00',
            (True, False): '10',
            (False, True): '01'
        })[x]
    def serialize_set(x):
        return '|'.join(serialize_tuple(t) for t in x)

    return ({serialize_set(a): b for a, b in result_statistics.items()}, bleu_score,
            {
                'neurons': neurons,
                'sample_values': sample_points
            })

if __name__ == '__main__':

    import spacy
    from spacy.tokens import Doc

    en = spacy.load('en')
    es = spacy.load('es')

    # Simple Spanish taggers:
    # =======================
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

    def make_tagger_function(wanted_pos, match_set):
        def tagger_function(tokens):
            return [pos == wanted_pos and
                all(key in morph and
                        morph[key] in match_set[key]
                        for key in match_set)
                for pos, morph in tag_es(tokens)]
        return tagger_function

    tag_past = make_tagger_function('VERB', {'VerbForm': ('Fin',), 'Tense': ('Past', 'Imp')})
    tag_present = make_tagger_function('VERB', {'VerbForm': ('Fin',), 'Tense': ('Pres',)})
    tag_singular = make_tagger_function('NOUN', {'Number': ('Sing',)})
    tag_plural = make_tagger_function('NOUN', {'Number': ('Plur',)})
    tag_masculine = make_tagger_function('NOUN', {'Gender': ('Masc',)})
    tag_feminine = make_tagger_function('NOUN', {'Gender': ('Fem',)})

    # English domain taggers:
    # =======================
    GENDER_ALTERNATING_WORDS = [
        # From "Gender Bias in Neural Natural Language Processing" (Lu, et al.)
        'gods','goddesses','nephew','niece','baron','baroness','father',
        'mother','dukes','duchesses','dad','mom','beau','belle','beaus','belles',
        'daddies','mummies','policeman','policewoman','grandfather','grandmother',
        'landlord','landlady','landlords','landladies','monks', 'nuns''stepson',
        'stepdaughter','milkmen','milkmaids','chairmen','chairwomen','stewards',
        'stewardesses','masseurs','masseuses','son-in-law',
        'daughter-in-law','priests','priestesses','steward','stewardess','emperor',
        'empress','son','daughter''kings','queens','proprietor','proprietres',
        'grooms','brides','gentleman','lady','king','queen','governor','matron',
        'waiters','waitresses','daddy','mummy''emperors','empresses','sir','madam',
        'wizards','witches','sorcerer','sorceress','lad','lass','milkman','milkmaid'
        'grandson','granddaughter','congressmen','congresswomen','dads','moms'

        # Added by hand
        'congressman', 'congresswoman', 'chairman', 'chairwoman',
        'chair', 'cheif', 'president'
    ]

    def tag_verbs(source_tokens):
        doc = Doc(vocab=en.vocab, words=source_tokens)
        en.tagger(doc)

        return [i for i, t in enumerate(doc) if t.pos_ == 'VERB']


    def descendant_of_numeral(t):
        return t.pos_ == 'NUM' or t != t.head and descendant_of_numeral(t.head)
    def ancestor_of_numeral(t):
        return t.pos_ == 'NUM' or any(ancestor_of_numeral(c) for c in t.children)
    def related_to_numeral(t):
        return descendant_of_numeral(t) or ancestor_of_numeral(t)
    def tag_nonnumeric_nouns(source_tokens):
        doc = Doc(vocab=en.vocab, words=source_tokens)
        en.tagger(doc)
        en.parser(doc)

        return [i for i, t in enumerate(doc) if t.pos_ == 'NOUN' and
                not related_to_numeral(t)]

    def tag_gender_alternating_nouns(source_tokens):
        doc = Doc(vocab=en.vocab, words=source_tokens)
        en.tagger(doc)

        return [i for i, t in enumerate(doc) if t.pos_ == 'NOUN' and
                t.text.lower() in GENDER_ALTERNATING_WORDS]

    # General domain interpreter
    # ==========================
    def domain_interpret(domain_base,
            source, align,
            target_property, use_spread):

        # Filter out to only those words which are not already
        # the target and *are* aligned to some other tag
        filtered = [i
                for i in domain_base(source) if
                any(x[target_property] for x in align[i][1]) and not
                any(x[1 - target_property] for x in align[i][1])]

        # If use_spread is enabled, use the parse tree
        if use_spread:
            doc = Doc(vocab=en.vocab, words=source)
            en.parser(doc)

            # Modify all dependents also
            return [
                (j, tuple(i for i, d in enumerate(doc)
                    if d == doc[j] or d.head == doc[j]))
                for j in filtered
            ]

        else:
            # Just modify this word
            return [(i, (i,)) for i in filtered]

    # Tokenized dataset loader
    # =======================
    def load_tokenized(fname):
        result = []
        with open(fname) as f:
            for line in f:
                result.append(line.strip().split(' '))
        return result

    def run_experiment(experiment_type = 'tense',
            target_property = 1,
            magnitude = 1,
            selection_strategy = 'target',
            num_neurons = 1,
            use_spread = False):

        # Everybody uses this model
        model = 'models-brnn/en-es-1.pt'

        searching_corpus_source = 'un-data/test/en'
        searching_corpus_target = 'output/en-es-1-brnn.txt'
        searching_dump = 'layer-dump/en-es-1-brnn.dump.pt'

        alignment_training_source = 'data/UNv1.0.6way.en.tok.2m'
        alignment_training_target = 'data/UNv1.0.6way.es.tok.2m'

        if experiment_type == 'tense':
            property_taggers = [tag_past, tag_present]
            source = 'un-data/test/en'
            target = 'output/en-es-1-brnn.txt'
            gold = 'un-data/test/es'
            domain_base = tag_verbs

        elif experiment_type == 'number':
            property_taggers = [tag_plural, tag_singular]
            source = 'un-data/test/en'
            target = 'output/en-es-1-brnn.txt'
            gold = 'un-data/test/es'
            domain_base = tag_nonnumeric_nouns

        elif experiment_type == 'gender':
            property_taggers = [tag_masculine, tag_feminine]
            source = 'data/gender-%d-en.tok' % target_property
            target = 'output/en-es-1-gender-%d.txt' % target_property
            gold = 'data/gender-%d-es.tok' % target_property
            domain_base = tag_gender_alternating_nouns

        result = single_target_modification(
            model = model,
            source = load_tokenized(source),
            target = load_tokenized(target),
            gold = load_tokenized(gold),

            alignment_training_source = load_tokenized(alignment_training_source),
            alignment_training_target = load_tokenized(alignment_training_target),

            searching_corpus_source = load_tokenized(searching_corpus_source),
            searching_corpus_target = load_tokenized(searching_corpus_target),
            searching_dump = searching_dump,

            property_taggers = property_taggers,
            target_property = target_property,
            magnitude_cap = magnitude,

            selection_strategy = selection_strategy,
            num_neurons = num_neurons,

            domain = lambda x, a: domain_interpret(domain_base, x, a, target_property, use_spread)
        )

        with open('clean-results/%s-%d-%f-%s-%d-%r.json' % (experiment_type, target_property, magnitude, selection_strategy, num_neurons, use_spread), 'w') as f:
            json.dump(result, f)

    '''
    import argparse

    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('--experiment_type', type=str,
            help='Experiment type to run (gender/number/tense)')
    parser.add_argument('--target_property', type=int,
            help='Target property index (0/1)')
    parser.add_argument('--magnitude', type=float,
            help='Magnitude of modification')
    parser.add_argument('--selection_strategy', type=str,
            help='Selection strategy (target/all)')
    parser.add_argument('--num_neurons', type=int,
            help='Number of top neurons to take')
    parser.add_argument('--use_spread', action='store_true',
            help='Set to modify multiple words')

    args = parser.parse_args()

    run_experiment(args
    '''

    # Indexed version:
    import sys

    index = int(sys.argv[1])

    t = 0
    for experiment_type in ['gender', 'number', 'tense']:
        for target_property in [0, 1]:
            for magnitude in [1, 2 ** 0.5, 2, 2 ** 1.5, 4, 2 ** 2.5, 8, 2 ** 3.5, 16, 2 ** 4.5,
                    32, 64, 128]:
                for selection_strategy in ['target', 'all']:
                    for num_neurons in [1, 2, 4, 8]:
                        if t == index:
                            print('EXPERIMENT RUNNING:', (experiment_type, target_property,
                                magnitude, selection_strategy, num_neurons))
                            run_experiment(experiment_type,
                                    target_property,
                                    magnitude,
                                    selection_strategy,
                                    num_neurons)
                        t += 1
