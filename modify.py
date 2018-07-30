#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

import sys

import torch

from onmt.utils.misc import get_logger
from onmt.translate.translator import build_translator

from tqdm import tqdm

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts
import json
import io

import os

class FakeOpt(object):
    def __init__(self,
            beam_size=None,
            n_best=None,
            max_best=None,
            min_length=None,
            max_length=None,
            stepwise_penalty=None,
            block_ngram_repeat=None,
            ignore_when_blocking=None,
            dump_beam=None,
            dump_layers=None,
            report_bleu=None,
            model=None,
            batch_size=None,
            data_type=None,
            replace_unk=None,
            alpha=None,
            beta=None,
            length_penalty=None,
            coverage_penalty=None,
            gpu=None,
            verbose=None):
        self.alpha = alpha
        self.beta = beta
        self.length_penalty = length_penalty
        self.coverage_penalty = coverage_penalty
        self.beam_size = beam_size
        self.n_best = n_best
        self.max_best = max_best
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


def run_modifications(
        corpus=None,
        neuron=None,
        layer=None,
        means=None,
        model=None):

    opt = FakeOpt(
        beam_size=5,
        min_length=10,
        max_length=100,
        stepwise_penalty=False,
        block_ngram_repeat=0,
        ignore_when_blocking=[],
        replace_unk=True,
        model=model,
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
        data_type='text'
    )

    translator = build_translator(opt, report_score=False, logger=get_logger(), use_output=False)

    toggles = [x[2] for x in corpus]
    sources = [x[0] for x in corpus]

    def intervene(layer_data, sentence_index, index):
        if index == layer:
            tqdm.write('Flipping (%d, %d) to %f' % (toggles[sentence_index], neuron, means[1 - corpus[sentence_index][3]]))
            layer_data[toggles[sentence_index]][0][neuron] =\
                    means[1 - corpus[sentence_index][3]]
        return layer_data

    modified = []
    for i, source in enumerate(tqdm(sources)):
        stream = io.StringIO()
        tqdm.write('Attempting to change the class of %s' % (source.split(' ')[toggles[i]],))
        translator.translate(src_data_iter=[source],
                             src_dir='',
                             batch_size=1,
                             attn_debug=False,
                             intervention=lambda l, j: intervene(l, i, j),
                             out_file=stream)
        translation = stream.getvalue()
        tqdm.write(translation)
        sys.stdout.flush()
        modified.append(translation)
    return list(zip(sources, modified))
