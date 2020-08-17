"""
Script to compare the runtime of the LSH and the alergia heuristic
"""


import argparse
import glob
import inspect
import os
import subprocess
import sys

import numpy as np
# import graphviz as graphviz
import openpyxl
from IPython.display import Image, display

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from pypackage.flexfringe.eval import profiling_evaluation, get_selectivity_threshold, error_based_classification, \
    fingerprint_based_classification, get_evaluation_metrics, get_profiling_threshold, evaluate_PAC, \
    calculate_perplexity


def flexfringe(*args, **kwargs):
    command = ["--help"]

    if (len(kwargs) > 1):
        command = ["--satdfabound=200"]
        for key in kwargs:
            command += ["-" + key + "=" + kwargs[key]]

    proc = subprocess.Popen(["../flexfringe", ] + command + [args[0]], stdout=subprocess.PIPE)
    output = proc.stdout.read()
    runtime, states = 0, 'too many'
    for line in output.decode("utf-8").split('\n'):
        print(line)
        if line.startswith('found intermediate solution with'):
            states = int(line.split()[4])
        if line.startswith('Run time:'):
            runtime = line.split()[-1].rstrip('[s]')
    return states, runtime



def train_model(malicious_traces_file, difference, arguments):
    heuristic = arguments[0]
    data = arguments[1]

    output_dir = malicious_traces_file + '_dir_{}/'.format(heuristic)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-M', help="batch or stream depending on the mode of operation", type=str,
                        default='batch')
    parser.add_argument('--heuristic_name', '-hn', help="The Name of the heuristic", type=str, default='lsh4')
    parser.add_argument('--data_name', '-d', help="Type of data for the heuristic", type=str, default='lsh4_data')
    parser.add_argument('--extrapar', '-p', help="extrapar", type=str, default='0.05')
    parser.add_argument('--state_count', '-q', help="state_count", type=str, default='10')
    parser.add_argument('--symbol_count', '-y', help="symbol_count", type=str, default='10')
    parser.add_argument('--satdfabound', '-D', help="satdfabound", type=str, default='200')
    # parser.add_argument('--delta', '-D', help="delta", type=str, default='0.95')
    parser.add_argument('--epsilon', '-e', help="epsilon", type=str, default='0.3')
    parser.add_argument('--seed', '-s', help="seed", type=str, default='42')
    parser.add_argument('--testmerge', '-t', help="testmerge", type=str, default='0')
    parser.add_argument('--klthreshold', '-T', help="klthreshold", type=str, default='0.01')
    parser.add_argument('--largestblue', '-a', help="largestblue", type=str, default='0')
    parser.add_argument('--finalred', '-f', help="finalred", type=str, default='0')
    parser.add_argument('--tablesize', '-S', help="tablesize", type=str, default='1000')
    parser.add_argument('--numoftables', '-N', help="numoftables", type=str, default='1')
    parser.add_argument('--vectordimension', '-v', help="vectordimension", type=str,
                        default='100')  # exeprioments done with 50
    parser.add_argument('--binarycodebytes', '-c', help="binarycodebytes", type=str, default='100')
    parser.add_argument('--difference', '-i', help="difference", type=str, default='20')
    parser.add_argument('--width', '-W', help="bin width", type=str, default='70')
    parser.add_argument('--index', '-g', help="Index mode, you can choose 1(CAUCHY) or 2(GAUSSIAN)", type=str,
                        default='1')
    parser.add_argument('--output_dir', '-o', help="output-dir", type=str, default=output_dir)
    parser.add_argument('--train_file', '-trf', help="Train file", type=str,
                        default=malicious_traces_file)

    # parser.add_argument('--dot_file', '-df', help="Dot file", type=str,
    #                     default=dot_file)

    parser.add_argument('--test_file', '-tsf', help="Test file for SPICE", type=str,
                        default='data/PAutomaC-competition_sets/11.pautomac.test')

    parser.add_argument('--prefix_file', '-pf', help="Prefix file for SPICE", type=str,
                        default='data/SPiCe_Offline/prefixes/14.spice.prefix.public')

    parser.add_argument('--target_file', '-tf', help="Target file for SPICE", type=str,
                        default='data/SPiCe_Offline/targets/14.spice.target.public')

    args = parser.parse_args()

    states, runtime = flexfringe(args.train_file, M=args.mode, h=heuristic, d=data, q=args.state_count,
                                 y=args.symbol_count, a=args.largestblue, f=args.finalred,
                                 # D=args.satdfabound,
                                 e=args.epsilon,
                                 p=args.extrapar,
                                 s=args.seed, t=args.testmerge, T=args.klthreshold, o=args.output_dir,
                                 i=difference, c=args.binarycodebytes, v=args.vectordimension, S=args.tablesize,
                                 N=args.numoftables, W=args.width, g=args.index)
    return states, runtime


def get_optimal_PAC_scores():
    # #   MY CODE
    data_dir = '../data/PAutomaC-competition_sets'
    all_trace_files = glob.glob(data_dir + "/*train")
    all_trace_files = [t.replace('\\', '/') for t in all_trace_files]
    all_trace_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    optimal_perplexity_scores = {}
    for trace_file in all_trace_files:  # missing 22,takes toooo much time
        scenario = trace_file.split('/')[-1].split('.')[0]

        tst = trace_file.replace('train', 'test')
        target_file = tst.rstrip('.test') + '_solution.txt'

        with open(target_file, 'r') as t:
            target_values = t.readlines()[1:]
        target_values = [float(t) for t in target_values]
        optimal_perplexity_score = calculate_perplexity(target_values, target_values)
        print('Scenario {}: {}'.format(scenario, optimal_perplexity_score))
        optimal_perplexity_scores[int(scenario)] = optimal_perplexity_score
    return optimal_perplexity_scores



def get_PAC_scores():
    runtime_alergia, states_alergia, perplexity_score_alergia = 0, 0, 0
    runtime_lsh, states_lsh, perplexity_score_lsh = 0, 0, 0
    data_dir = '../data/PAutomaC-competition_sets'
    all_trace_files = glob.glob(data_dir + "/*train")
    all_trace_files = [t.replace('\\', '/') for t in all_trace_files]
    all_trace_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    arguments_lsh = ['lsh4', 'lsh4_data']
    arguments_alergia = ['alergia', 'alergia_data']
    # with open('0_20_time_results_0dot01.txt', 'w') as fout:
    for trace_file in all_trace_files:  # missing 22,takes toooo much time
        scenario = trace_file.split('/')[-1].split('.')[0]
        with open('{}_time_results_0dot01.txt'.format(scenario), 'w') as fout:
            with open(trace_file, 'r') as f:
                i = f.readline().split()[1]
            states_lsh, runtime_lsh = train_model(trace_file, str(int(i)), arguments_lsh)
            dot_file = trace_file.replace('\\', '/') + '_dir_lsh4/final.dot'
            perplexity_score_lsh, optimal_perplexity_score = evaluate_PAC(dot_file, trace_file.replace('train', 'test'))

            with open(trace_file, 'r') as f:
                i = f.readline().split()[1]
            states_alergia, runtime_alergia = train_model(trace_file, str(int(i)), arguments_alergia)
            dot_file = trace_file.replace('\\', '/') + '_dir_alergia/final.dot'
            perplexity_score_alergia, optimal_perplexity_score = evaluate_PAC(dot_file,
                                                                              trace_file.replace('train', 'test'))

            print("perplexity_score_lsh: ", perplexity_score_lsh)
            print("perplexity_score_alergia: ", perplexity_score_alergia)
            print("optimal_perplexity_score: ", optimal_perplexity_score)

            runfile = trace_file.replace('\\', '/').split('/')[-1]
            fout.write('File: {}\n 1-SD LSH: {}, {}, {:.3f} \n ALERGIA: {}, {}, {:.3f}\n\n'.format(runfile,
                                                                                                   runtime_lsh,
                                                                                                   states_lsh,
                                                                                                   perplexity_score_lsh,
                                                                                                   runtime_alergia,
                                                                                                   states_alergia,
                                                                                                   perplexity_score_alergia))


def read_scores():
    lsh_results = {}
    alergia_results = {}
    optimal_scores = get_optimal_PAC_scores()
    for s in range(1, 49):
        f = currentdir.replace('\\', '/') + '/{}_time_results_0dot01.txt'.format(s)
        try:
            with open(f, 'r') as fin:
                lines = fin.readlines()
                lsh_results[s] = [float(v.strip()) for v in lines[1].split(':')[1].split(',')]
                alergia_results[s] = [float(v.strip()) for v in lines[2].split(':')[1].split(',')]
        except FileNotFoundError:
            pass

    lsh_times = []
    alergia_times = []

    lsh_error = []
    alergia_error = []

    lsh_states = []
    alergia_states = []

    for s in range(1, 49):
        lsh_times += [lsh_results[s][0]]
        lsh_states += [lsh_results[s][1]]
        lsh_error += [lsh_results[s][2] - optimal_scores[s]]

        alergia_times += [alergia_results[s][0]]
        alergia_states += [alergia_results[s][1]]
        alergia_error += [alergia_results[s][2] - optimal_scores[s]]
    print('LSH mean time: {}'.format(np.mean(lsh_times)))
    print('LSH mean states: {}'.format(np.mean(lsh_states)))
    print('LSH mean error: {}'.format(np.mean(lsh_error)))

    print('ALERGIA mean time: {}'.format(np.mean(alergia_times)))
    print('ALERGIA mean states: {}'.format(np.mean(alergia_states)))
    print('ALERGIA mean error: {}'.format(np.mean(alergia_error)))

    # print('OLA KALA')


if __name__ == '__main__':
    # get_optimal_PAC_scores()
    read_scores()
    # get_PAC_scores()