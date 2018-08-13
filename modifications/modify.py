#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

import sys

import torch

from onmt.utils.misc import get_logger
from onmt.translate.translator import build_translator

from tqdm import tqdm

from modifications.util import file_cached_function

import json
import io

import os

class FakeOpt(object):
    def __init__(self,
            beam_size=5,
            min_length=10,
            max_length=100,
            stepwise_penalty=False,
            block_ngram_repeat=0,
            ignore_when_blocking=[],
            replace_unk=True,
            model=None,
            verbose=False,
            report_bleu=False,
            batch_size=30,
            n_best=1,
            dump_beam='',
            dump_layers='',
            gpu=0,
            alpha=0,
            beta=0,
            length_penalty='none',
            coverage_penalty='none',
            data_type='text'):
        self.alpha = alpha
        self.beta = beta
        self.length_penalty = length_penalty
        self.coverage_penalty = coverage_penalty
        self.beam_size = beam_size
        self.n_best = n_best
        self.max_length = max_length
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self.dump_beam = dump_beam
        self.dump_layers = dump_layers
        self.report_bleu = report_bleu
        self.data_type = data_type
        self.replace_unk = replace_unk
        self.batch_size = batch_size
        self.gpu = gpu
        self.verbose = verbose
        self.model = model


def _modify(
        corpus=None,
        neurons=None,
        values=None,
        model=None):

    opt = FakeOpt(model=model)

    translator = build_translator(opt, report_score=False, logger=get_logger(), use_output=False)

    sources, toggles = zip(*corpus)
    _, toggles = zip(*toggles)

    def intervene(layer_data, sentence_index, index):
        for (layer, neuron), value in zip(neurons, values):
            if index == layer:
                for i in toggles[sentence_index]:
                    layer_data[i][0][neuron] = value
        return layer_data

    modified = []
    for i, source in enumerate(tqdm(sources)):
        stream = io.StringIO()

        # Logging:
        print(toggles[i])
        tqdm.write('Source: %s' % ' '.join(source))
        tqdm.write('Target: %s' % ' '.join(source[j] for j in toggles[i]))

        translator.translate(src_data_iter=[' '.join(source)],
                             src_dir='',
                             batch_size=1,
                             attn_debug=False,
                             intervention=lambda l, j: intervene(l, i, j),
                             out_file=stream)
        translation = stream.getvalue()

        # Logging:
        tqdm.write('Result: %s' % translation)

        modified.append(translation.strip().split(' '))
    return modified

modify = file_cached_function(_modify, 1)
