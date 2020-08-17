#!/usr/bin/python3
import inspect
import json
import os
import re
from scipy.stats import rv_discrete
import sys
import numpy as np
from sys import *
import math
import string
from random import shuffle

# we should define these somewhere that is associated with the
# evaluation_functino. Maybe member variables that are initialized in
# the print_dot function ... but 
# that would make evaluation depend on
# the output format, which is not very elegant.
# technically we should use the apta after merging, and
MEAN_REGEX = '(?P<state>\d+) \[shape=(doublecircle|circle|ellipse) label=\"\[(?P<mean>\d+)\].*\"\];'
SYMLST_REGEX = "((?P<sym>\d+):(?P<occ>\d+))+"
TRAN_REGEX = "(?P<sst>.+) -> (?P<dst>.+) \[label=\"(?P<slst>(.+))\"[ style=dotted]*  \];$"


# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

def load_means(content):
    means = []

    for line in content.split("\n"):
        matcher = re.match(MEAN_REGEX, line.strip())
        #      print(line.strip())
        #      print(MEAN_REGEX)
        if matcher is None: continue
        # for match in matcher:
        match = matcher
        if match.group("mean") == "0": continue
        means.append("{} -> {} [label=\" {} [{}:0]\"];".format(match.group("state"), match.group("state"),
                                                               "-1", match.group("mean")))
    return means


def load_model_from_file(dot_path, normalize=False):
    with open(dot_path, "r") as f:
        return load_model(f.read(), normalize)


# Read and extract automaton structure from dot file
# returns: dictionary dict[currect_state][symbol_read] = next_state
def load_model(dot_string, normalize=False, with_means=False):
    model = {}
    if with_means:
        means = "\n".join(load_means(dot_string))
        dot_string = means + dot_string

    for line in dot_string.split('\n'):
        matcher = re.match(TRAN_REGEX, line.strip())
        # print(line + " found" + '\n')
        if matcher is not None:
            # print("found match")
            sstate = matcher.group("sst")
            dstate = matcher.group("dst")
            symlist = matcher.group("slst")
            if sstate not in model:
                model[sstate] = {}
            for smatcher in re.finditer(SYMLST_REGEX, symlist):
                symbol = int(smatcher.group("sym"))
                occurrences = int(smatcher.group("occ"))
                # occurrences = 1
                if symbol not in model[sstate]:
                    model[sstate][symbol] = (dstate, occurrences)
                else:
                    gamma = 0

    # normalizing occurrence counts to probabilities
    if normalize:
        for state in model:
            ssum = sum(map(lambda x: x[1], model[state].values()))
            for symbol in model[state]:
                dstate, occurrences = model[state][symbol]
                model[state][symbol] = (dstate, occurrences / float(ssum))
    return model


# given model, make predictions for all lines in test-path and
# write result to pred-path
def get_word_acceptance(model, test_array):
    REJECT = 0
    ACCEPT = 1

    accept = 0
    waccept = 0
    reject = 0
    okwindows = 0
    wneg = 0
    pred_array = []
    for line in test_array:
        state = "0"

        symbol_sequence = []

        # for actual_symbol in sequence:
        for i in range(1, len(line.strip().split(" "))):
            symbol_sequence.append(int(line.strip().split(" ")[i].split(",")[0]))

        # print symbol_sequence

        # loop over word
        rejected = False
        for i, actual_symbol in enumerate(symbol_sequence):
            if state not in model:
                pred_array.append(REJECT)
                rejected = True
                break
            else:
                try:
                    state = model[state][actual_symbol][0]
                except KeyError:
                    rejected = True
                    pred_array.append(REJECT)
                    # print "in state %s read %s, no transition" % (state, actual_symbol)
                    # print model[state]
                    reject += 1
                    break

        if not rejected:
            pred_array.append(ACCEPT)
            # print "Accepted %s" % (accept+reject)
            accept += 1
            waccept += 1

        else:
            gamma = 0
            # print "Rejected"

        if ((accept + reject) % 5) == 0:
            if waccept > 3:
                #             pred_array.append("windowok")
                # print "Window ok"
                okwindows += 1
            else:
                if waccept == 0:
                    wneg += 1
            waccept = 0

    if accept == 0:
        # print "no acceptance"
        accept = 1
    if reject == 0:
        # print "no reject"
        reject = 1

    # this output is for the CNSM/LCN papers
    #     if float(accept)/(accept+reject) > 0.75:
    #         print("BOTNET")
    #     else:
    #         print("CLEAN")

    # this return should probably an array of (0/1, prob) for the
    # tst input argument instead of putting it into the output file
    print("Acc %s Rej %s, total %s, ratio %s, quota %s, Windows: %s of %s, %s neg" % (
        accept, reject, float(accept) / (accept + reject), float(accept) / reject, float(reject) / accept, okwindows,
        (accept + reject) / 5, wneg))

    return pred_array


# given model, make predictions for all lines in test-path and
# write result to pred-path
def get_word_acceptance_prob(model, test_array, length):
    with open("my_solution.txt", "w") as fout:

        fout.write(str(length) + "\n")
        prob_list = []
        for line in test_array:
            state = "0"
            prob = 1

            symbol_sequence = []

            # for actual_symbol in sequence:
            for j in range(1, len(line.strip().split(" "))):
                symbol_sequence.append(int(line.strip().split(" ")[j].split(",")[0]))
            # symbol_sequence = list(map(int, line.strip().split()))[1:]
            # print symbol_sequence

            # loop over word
            for i, actual_symbol in enumerate(symbol_sequence):

                if state not in model:
                    prob = 1e-5
                    # prob = 0
                    # fout.write(str(prob) + "\n")
                    # break
                else:
                    try:
                        prob *= model[state][actual_symbol][1]
                        state = model[state][actual_symbol][0]
                    except KeyError:
                        # prob = 0
                        prob = 1e-5
                        # fout.write(str(prob) + "\n")
                        # break
                        # print "in state %s read %s, no transition" % (state, actual_symbol)
                        # print model[state]
            fout.write(str(prob) + "\n")
            prob_list += [prob]

    total_prob = np.sum(prob_list)
    prob_list = [p / total_prob for p in prob_list]
    return prob_list


def calculate_perplexity(predicted_values, target_values):
    # with open(predicted_file, 'r') as p, open(target_file, 'r') as t:
    #     predicted_values = p.readlines()[1:]
    #     target_values = t.readlines()[1:]
    #
    # predicted_values = [float(p) for p in predicted_values]
    # target_values = [float(t) for t in target_values]

    s = 0
    for p, t in zip(predicted_values, target_values):
        s += t * np.log2(p)
    score = pow(2, -s)
    return score


def predict(prefix, model):
    # traverse
    state = "0"
    for s in prefix.split(" "):
        if s == "": continue
        try:
            state = model[state][int(s)][0]
        #      print(state)
        except:
            #      print("hello")
            break
    try:
        if len(model[state].items()) == 0:
            return [(-1, (0, 1))]
    except KeyError:
        return [(-1, (0, 1))]
    return sorted(model[state].items(), key=lambda x: x[1][1], reverse=True)


def spice_rankings(prefix_file, dot="../../dfafinal.dot"):
    model = load_model_from_file(dot, normalize=True)

    with open(prefix_file) as pf:
        prefixes = pf.readlines()
    seq_not_found = 0
    all_rankings = []
    for pr in prefixes[1:]:
        rankings = []
        prefix = pr[1:]
        state = "0"
        for i, s in enumerate(prefix.split()):
            if i == len(prefix.split()) - 1:
                try:
                    next_symbols = model[state]
                except KeyError:
                    print("State not found in the model")
                    raise
                symbols_list = []
                for symbol, value in next_symbols.items():
                    symbols_list += [(symbol, value[1])]
                symbols_list = sorted(symbols_list, key=lambda x: x[1], reverse=True)
                for symbol in symbols_list[:5]:
                    rankings += [symbol[0]]
                break
            else:
                try:
                    state = model[state][int(s)][0]
                except KeyError:
                    # print("State or symbol not found in the model")
                    seq_not_found += 1
                    num_of_symbols = int(prefixes[0].split()[1])
                    for t in range(num_of_symbols):
                        rankings += [t]
                    # raise
                    shuffle(rankings)
                    break
        all_rankings += [rankings[:5]]
    print("Sequences not found: ", seq_not_found)
    return all_rankings


def evaluate_PAC(dot, tst):
    # you can also invoke the script from the command line

    m = load_model_from_file(dot, normalize=True)

    # print(m)
    # with open('model.json', 'w') as f:
    #     json.dump(m, f, indent=4, separators=(',', ': '))

    with open(tst, "r") as fh:
        test = fh.readlines()

    with open("resultfile.txt", 'w') as fh:
        res = []
        for line in test[1:]:
            sum = ""
            for elm in predict(line, m):
                sum += str(elm[0]) + " "
            res.append(sum + '\n')
        fh.writelines(res)

    # #   MY CODE
    with open(tst) as fh:
        rs = fh.readlines()
    pred_array = get_word_acceptance(m, rs[1:])
    prob_list = get_word_acceptance_prob(m, rs[1:], rs[0].split()[0])
    target_file = tst.rstrip('.test') + '_solution.txt'
    # target_file = '../../data/PAutomaC-competition_sets/2.pautomac_solution.txt'

    with open(target_file, 'r') as t:
        target_values = t.readlines()[1:]
    target_values = [float(t) for t in target_values]

    perplexity_score = calculate_perplexity(prob_list, target_values)
    optimal_perplexity_score = calculate_perplexity(target_values, target_values)

    # print("Perplexity score: ", perplexity_score)
    return perplexity_score, optimal_perplexity_score
    # for i, p in enumerate(pred_array):
    #     if p == 0:
    #         print(i)
    # print(pred_array)


if __name__ == "__main__":
    # evaluate_PAC()
    spice_rankings("../../data/SPiCe_Offline/prefixes/1.spice.prefix.public", "../../dfafinal.dot")


# def test_all():
#     # you can also invoke the script from the command line
#     if len(sys.argv) > 1:
#         dot = sys.argv[1]
#         tst = sys.argv[2]
#         of = sys.argv[3]
#     else:
#         print("usesage: ./name dotfile testfile outputfile")
#
#     m = load_model_from_file(dot, normalize=True)
#
#     data_dir = "../../data/PAutomaC-competition_sets"
#     for file in os.listdir(data_dir):
#         if file.endswith(".train"):
#             import subprocess
#             process = subprocess.Popen(
#                 "flexfringe --heuristic-name lsh --data-name lsh_data --state_count 10 --symbol_count 4 --satdfabound 100 --sinkson 0  --klthreshold 0.3 {}".format(
#                     trn), shell=True, stdout=subprocess.PIPE)
#             process.wait()
#             print(process.returncode)
#         if file.endswith(".test"):
#             tst = os.path.join(data_dir, file)
#         with open(tst) as fh:
#             rs = fh.readlines()[1:]
#         pred_array = get_word_acceptance(m, rs)
#         # for i, p in enumerate(pred_array):
#         #     if p == 0:
#         #         print(i)
#         # print(pred_array)

def find_proba(letter, target):
    for i in range(len(target)):
        if target[i] == letter:
            return float(target[i + 1])
    return 0


def evaluate_spice(rankings, targets_file):
    t = open(targets_file, "r")

    score = 0
    nb_prefixes = 0
    for i, ts in enumerate(t.readlines()):
        nb_prefixes += 1
        target = ts.split()
        ranking = rankings[i]

        denominator = float(target[0])
        prefix_score = 0
        k = 1
        for elmnt in ranking:
            if k == 1:
                seen = [elmnt]
                p = find_proba(elmnt, target)
                prefix_score += p / math.log(k + 1, 2)
            elif elmnt not in seen:
                p = find_proba(elmnt, target)
                prefix_score += p / math.log(k + 1, 2)
                seen = seen + [elmnt]
            k += 1
            if k > 5:
                break
        # print(nb_prefixes, su)
        score += prefix_score / denominator
    final_score = score / nb_prefixes
    print("Spice score: ", final_score)
    t.close()


def fingerprint_based_classification(malicious_model_file, candidate_traces_file, malicious_traces_file,
                                     benign_traces_files):
    if isinstance(malicious_model_file, dict):
        malicious_model = malicious_model_file
    else:
        malicious_model = load_model_from_file(malicious_model_file, normalize=True)
    with open(candidate_traces_file, 'r') as f:
        candidate_traces = f.readlines()
    with open(malicious_traces_file, 'r') as f:
        malicious_traces = f.readlines()
    if isinstance(list(malicious_model.keys())[0], tuple):
        initial_state = (-1, -1)
    else:
        initial_state = "0"
    malicious_counter = {}
    candidate_counter = {}

    for trace in malicious_traces:
        state = initial_state
        symbol_sequence = list(map(int, trace.strip().split()))[1:]
        for i, actual_symbol in enumerate(symbol_sequence):
            try:
                state = malicious_model[state][actual_symbol][0]
                if state in malicious_counter:
                    malicious_counter[state] += 1
                else:
                    malicious_counter[state] = 1
            except KeyError:
                break

    for trace in candidate_traces:
        state = initial_state
        symbol_sequence = list(map(int, trace.strip().split()))[1:]

        for i, actual_symbol in enumerate(symbol_sequence):
            try:
                state = malicious_model[state][actual_symbol][0]
                if state in candidate_counter:
                    candidate_counter[state] += 1
                else:
                    candidate_counter[state] = 1
            except KeyError:
                break

    benign_counter = {}
    for benign_traces_file in benign_traces_files:
        with open(benign_traces_file, 'r') as f:
            benign_traces = f.readlines()
        for trace in benign_traces:
            state = initial_state
            symbol_sequence = list(map(int, trace.strip().split()))[1:]

            for i, actual_symbol in enumerate(symbol_sequence):
                try:
                    state = malicious_model[state][actual_symbol][0]
                    if state in benign_counter:
                        benign_counter[state] += 1
                    else:
                        benign_counter[state] = 1
                except KeyError:
                    break

    all_states = set(list(candidate_counter.keys()) + list(malicious_counter.keys()) + list(benign_counter.keys()))
    error = 0
    for state in all_states:
        if state in candidate_counter and state in malicious_counter and state not in benign_counter:
            return 1
    return 0


def error_based_classification(malicious_model_file, candidate_traces_file, malicious_traces_file, threshold):
    if isinstance(malicious_model_file, dict):
        malicious_model = malicious_model_file
    else:
        malicious_model = load_model_from_file(malicious_model_file, normalize=True)
    with open(candidate_traces_file, 'r') as f:
        candidate_traces = f.readlines()
    with open(malicious_traces_file, 'r') as f:
        malicious_traces = f.readlines()
    if isinstance(list(malicious_model.keys())[0], tuple):
        initial_state = (-1, -1)
    else:
        initial_state = "0"
    malicious_counter = {}
    candidate_counter = {}
    total_malicious = 0
    total_candidates = 0
    for trace in malicious_traces:
        state = initial_state
        symbol_sequence = list(map(int, trace.strip().split()))[1:]
        for i, actual_symbol in enumerate(symbol_sequence):
            try:
                state = malicious_model[state][actual_symbol][0]
                total_malicious += 1
                if state in malicious_counter:
                    malicious_counter[state] += 1
                else:
                    malicious_counter[state] = 1
            except KeyError:
                break

    for trace in candidate_traces:
        state = initial_state
        symbol_sequence = list(map(int, trace.strip().split()))[1:]
        for i, actual_symbol in enumerate(symbol_sequence):
            try:
                state = malicious_model[state][actual_symbol][0]
                total_candidates += 1
                if state in candidate_counter:
                    candidate_counter[state] += 1
                else:
                    candidate_counter[state] = 1
            except KeyError:
                break
    all_states = set(list(candidate_counter.keys()) + list(malicious_counter.keys()))
    error = 0
    total = 0
    all_errors = []
    for state in all_states:
        if state in candidate_counter and state in malicious_counter:
            c_sum = candidate_counter[state]
            m_sum = malicious_counter[state]
        elif state in candidate_counter and state not in malicious_counter:
            c_sum = candidate_counter[state]
            m_sum = 0
        elif state not in candidate_counter and state in malicious_counter:
            c_sum = 0
            m_sum = malicious_counter[state]
        else:
            c_sum = 0
            m_sum = 0
        total += c_sum + m_sum
        error += np.abs(c_sum - m_sum)
        all_errors += [np.abs(c_sum - m_sum) / (c_sum + m_sum)]
    try:
        # if (np.mean(all_errors)) < threshold:
        if (float(error) / total) < threshold:
            return 1
        else:
            return 0
    except ZeroDivisionError:
        if 0 < threshold:
            return 1
        else:
            return 0


def profiling_evaluation(malicious_model_file, trace_file, highest_acceptance_ratio):
    if isinstance(malicious_model_file, dict):
        malicious_model = malicious_model_file
    else:
        malicious_model = load_model_from_file(malicious_model_file, normalize=True)
    if isinstance(list(malicious_model.keys())[0], tuple):
        initial_state = (-1, -1)
    else:
        initial_state = "0"
    fin = open(trace_file, "r")
    candidate_traces = fin.readlines()
    total_traces = int(candidate_traces[0].strip().split()[0])
    REJECT = 0
    ACCEPT = 1

    accept = 0
    # waccept = 0
    reject = 0
    # okwindows = 0
    # wneg = 0
    pred_array = []
    for trace_num, line in enumerate(candidate_traces[1:]):
        state = initial_state

        symbol_sequence = []

        # for actual_symbol in sequence:
        for i in range(1, len(line.strip().split(" "))):
            symbol_sequence.append(int(line.strip().split(" ")[i].split(",")[0]))

        # print symbol_sequence

        # loop over word
        rejected = False
        for i, actual_symbol in enumerate(symbol_sequence):
            if state not in malicious_model:
                pred_array.append(REJECT)
                rejected = True
                break
            else:
                try:
                    state = malicious_model[state][actual_symbol][0]
                except KeyError:
                    rejected = True
                    pred_array.append(REJECT)
                    # print "in state %s read %s, no transition" % (state, actual_symbol)
                    # print model[state]
                    reject += 1
                    break

        if not rejected:
            pred_array.append(ACCEPT)
            # print "Accepted %s" % (accept+reject)
            accept += 1
            # waccept += 1

        # if float(trace_num) >= float(0.25 * total_traces):
        # if trace_num >= math.floor(0.1*total_traces):
        if trace_num >= 25:
            # if reject + accept:
            #     acceptance_ratio = float(accept) / (accept + reject)
            if reject:
                acceptance_ratio = float(accept) / reject
            else:
                acceptance_ratio = 1
            if acceptance_ratio > highest_acceptance_ratio:
                return 1  # botnet
    #     if ((accept + reject) % 5) == 0:
    #         if waccept > 3:
    #             #             pred_array.append("windowok")
    #             # print "Window ok"
    #             okwindows += 1
    #         else:
    #             if waccept == 0:
    #                 wneg += 1
    #         waccept = 0
    #
    # if accept == 0:
    #     # print "no acceptance"
    #     accept = 1
    # if reject == 0:
    #     # print "no reject"
    #     reject = 1

    # # this output is for the CNSM/LCN papers
    #     if float(accept)/(accept+reject) > 0.75:
    #     if float(accept)/(reject) > 0.75:
    #         print("BOTNET")
    #     else:
    #         print("CLEAN")

    # this return should probably an array of (0/1, prob) for the
    # tst input argument instead of putting it into the output file
    # print("Acc %s Rej %s, total %s, ratio %s, quota %s, Windows: %s of %s, %s neg" % (
    #     accept, reject, float(accept) / (accept + reject), float(accept) / reject, float(reject) / accept, okwindows,
    #     (accept + reject) / 5, wneg))
    fin.close()
    return 0


def get_selectivity_threshold(malicious_model_file, benign_traces_files, malicious_traces_file):
    if isinstance(malicious_model_file, dict):
        malicious_model = malicious_model_file
    else:
        malicious_model = load_model_from_file(malicious_model_file, normalize=True)

    if isinstance(list(malicious_model.keys())[0], tuple):
        initial_state = (-1, -1)
    else:
        initial_state = "0"

    with open(malicious_traces_file, 'r') as f:
        malicious_traces = f.readlines()
    malicious_counter = {}
    total_malicious = 0
    for trace in malicious_traces:
        state = initial_state
        symbol_sequence = list(map(int, trace.strip().split()))[1:]
        # loop over word
        for i, actual_symbol in enumerate(symbol_sequence):
            try:
                state = malicious_model[state][actual_symbol][0]
                total_malicious += 1
                if state in malicious_counter:
                    malicious_counter[state] += 1
                else:
                    malicious_counter[state] = 1
            except KeyError:
                break
    all_errors = []
    for benign_traces_file in benign_traces_files:
        benign_counter = {}
        total_benign = 0
        with open(benign_traces_file, 'r') as f:
            benign_traces = f.readlines()
        for trace in benign_traces:
            state = initial_state
            symbol_sequence = list(map(int, trace.strip().split()))[1:]
            # print symbol_sequence
            # loop over word
            for i, actual_symbol in enumerate(symbol_sequence):
                try:
                    state = malicious_model[state][actual_symbol][0]
                    total_benign += 1
                    if state in benign_counter:
                        benign_counter[state] += 1
                    else:
                        benign_counter[state] = 1
                except KeyError:
                    break
        # for state, counter in benign_counter.items():
        #     benign_counter[state] = float(benign_counter[state]) / total_benign
        #
        # for state, counter in malicious_counter.items():
        #     malicious_counter[state] = float(malicious_counter[state]) / total_malicious

        all_states = set(list(benign_counter.keys()) + list(malicious_counter.keys()))
        error = 0
        total = 0
        for state in all_states:
            if state in benign_counter and state in malicious_counter:
                c_sum = benign_counter[state]
                m_sum = malicious_counter[state]
            elif state in benign_counter and state not in malicious_counter:
                c_sum = benign_counter[state]
                m_sum = 0
            elif state not in benign_counter and state in malicious_counter:
                c_sum = 0
                m_sum = malicious_counter[state]
            else:
                c_sum = 0
                m_sum = 0
            total += c_sum + m_sum
            error += np.abs(c_sum - m_sum)
            all_errors += [np.abs(c_sum - m_sum) / (c_sum + m_sum)]
    if total == 0:
        return 0
    else:
        return float(error) / total  # maybe do it as a percentage
        # return np.mean(all_errors)
    # return np.mean(all_errors) - np.std(all_errors)
    # return error  # maybe do it as a percentage


def get_evaluation_metrics(actual_labels, predicted_labels):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(predicted_labels)):
        if actual_labels[i] and predicted_labels[i]:
            TP += 1
        if not actual_labels[i] and predicted_labels[i]:
            FP += 1
        if actual_labels[i] and not predicted_labels[i]:
            FN += 1
        if not actual_labels[i] and not predicted_labels[i]:
            TN += 1

    # just in case that no TP or FP are found
    if not (TP + FP):
        precision = 1
    else:
        precision = TP / (TP + FP)
    if not (TP + FN):
        recall = 1
    else:
        recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FN + TN + FP)
    print('TP: ' + str(TP))
    print('FP: ' + str(FP))
    print('FN: ' + str(FN))
    print('TN: ' + str(TN))
    print('Accuracy: ' + str(accuracy))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    return TP, FP, TN, FN, accuracy, precision, recall


def get_profiling_threshold(malicious_model_file, configuration_trace_file):
    if isinstance(malicious_model_file, dict):
        malicious_model = malicious_model_file
    else:
        malicious_model = load_model_from_file(malicious_model_file, normalize=True)
    if isinstance(list(malicious_model.keys())[0], tuple):
        initial_state = (-1, -1)
    else:
        initial_state = "0"
    fin = open(configuration_trace_file, "r")
    configuration_traces = fin.readlines()
    total_traces = int(configuration_traces[0].strip().split()[0])
    REJECT = 0
    ACCEPT = 1

    accept = 0
    # waccept = 0
    reject = 0
    # okwindows = 0
    # wneg = 0
    pred_array = []
    highest_accepatnce_ratio = 0
    for trace_num, line in enumerate(configuration_traces[1:]):
        state = initial_state

        symbol_sequence = []

        # for actual_symbol in sequence:
        for i in range(1, len(line.strip().split(" "))):
            symbol_sequence.append(int(line.strip().split(" ")[i].split(",")[0]))

        # print symbol_sequence

        # loop over word
        rejected = False
        for i, actual_symbol in enumerate(symbol_sequence):
            if state not in malicious_model:
                pred_array.append(REJECT)
                rejected = True
                break
            else:
                try:
                    state = malicious_model[state][actual_symbol][0]
                except KeyError:
                    rejected = True
                    pred_array.append(REJECT)
                    # print "in state %s read %s, no transition" % (state, actual_symbol)
                    # print model[state]
                    reject += 1
                    break

        if not rejected:
            pred_array.append(ACCEPT)
            # print "Accepted %s" % (accept+reject)
            accept += 1
            # waccept += 1

        # if float(trace_num) >= float(0.25 * total_traces):
        if trace_num >= 25:
            # if reject + accept:
            #     acceptance_ratio = float(accept) / (accept + reject)
            if reject:
                acceptance_ratio = float(accept) / reject
            else:
                acceptance_ratio = 1
            if acceptance_ratio > highest_accepatnce_ratio:
                highest_accepatnce_ratio = acceptance_ratio
    fin.close()
    return highest_accepatnce_ratio
