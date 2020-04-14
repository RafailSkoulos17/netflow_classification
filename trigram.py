# -*- coding: utf-8 -*-
"""
    Created in Thu March  22 10:47:00 2016
    
    @author: Remi Eyraud & Sicco Verwer
    
    Usage: python 3gram_baseline.py train_file prefixes_file output_file
    Role: learn a 3-gram on the whole sequences of train_file, then generates a ranking of the 5 most probable symbols for each prefix of prefixes_file, stores these ranking in output_file (one ranking per line, in the same order than in the prefix file)
    Example: python 3gram_baseline.py ../train/0.spice.train ../prefixes/0.spice.prefix.public 0.spice.ranking
"""

from numpy import *
from decimal import *
from sys import *
import math


def train_trigram_model(malicious_traces_file):
    with open(malicious_traces_file, 'r') as f:
        malicious_traces = f.readlines()
    model = {}
    for sequence in malicious_traces:
        symbol_sequence = list(map(int, sequence.strip().split()))[1:]
        ngramseq = [-1, -1] + symbol_sequence + [-2]
        for start in range(len(ngramseq) - 2):
            end = start + 2
            if tuple(ngramseq[start:end]) in model:
                table = model[tuple(ngramseq[start:end])]
                if ngramseq[end] in table:
                    table[ngramseq[end]] = (table[ngramseq[end]][0], table[ngramseq[end]][1] + 1)
                else:
                    table[ngramseq[end]] = (tuple(ngramseq[start + 1:end + 1]), 1)
            else:
                table = {}
                table[ngramseq[end]] = (tuple(ngramseq[start + 1:end + 1]), 1)
                # table[-1] = 1
                model[tuple(ngramseq[start:end])] = table
    for state in model:
        ssum = sum(list(map(lambda x: x[1], model[state].values())))
        for symbol in model[state]:
            dstate, occurrences = model[state][symbol]
            model[state][symbol] = (dstate, occurrences / float(ssum))
    return model


if __name__=='__main__':
    print("Getting training sample")
    train_file = 'data/ctu_13/bidirectional/single_scenario/scenario_50/100-50_median__traces_ctu_50/median__traces_ctu_50_infected_147.32.84.209_traces'
    # alphabet, train = readset(open(train_file, "r"))
    print("Start Learning")
    model = train_trigram_model(train_file)
    print("Learning Ended")
    print(profiling_evaluation(model, train_file, 0.25))
