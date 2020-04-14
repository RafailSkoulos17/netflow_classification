#!/usr/bin/env python
# coding: utf-8
import os

import pandas as pd
from functools import reduce
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
from IPython.display import display, HTML
import warnings
import datetime
from pandas import Timestamp
import copy
import time
from sklearn.model_selection import train_test_split
import math
from copy import deepcopy
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# ## Functions used
# IPs for scenario 1-1
ip_dict = {}
ip_dict['1-1'] = '192.168.100.103'
ip_dict['3-1'] = '192.168.2.5'
ip_dict['7-1'] = '192.168.100.108'
ip_dict['8-1'] = '192.168.100.113'

ip_dict['9-1'] = '192.168.100.111'
ip_dict['17-1'] = '192.168.100.111'
ip_dict['20-1'] = '192.168.100.103'
ip_dict['21-1'] = '192.168.100.113'

ip_dict['33-1'] = '192.168.1.197'
ip_dict['34-1'] = '192.168.1.195'
ip_dict['35-1'] = '192.168.1.195'
ip_dict['36-1'] = '192.168.1.198'

ip_dict['39-1'] = '192.168.1.194'
ip_dict['42-1'] = '192.168.1.197'
ip_dict['43-1'] = '192.168.1.198'
ip_dict['44-1'] = '192.168.1.199'

ip_dict['48-1'] = '192.168.1.200'
ip_dict['49-1'] = '192.168.1.193'
ip_dict['52-1'] = '192.168.1.197'
ip_dict['60-1'] = '192.168.1.195'


def read_data(filepath):
    preprocess_data(filepath)
    # dateparse = lambda x: time.strftime('%Y-%m-%d %H:%M:%S:{}'.format(x % 1000), time.gmtime(x / 1000.0))
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    dateparse = lambda x: pd.to_datetime(x, unit='s')
    data = pd.read_csv(filepath + '_v2', delimiter=',', parse_dates=['ts'], date_parser=dateparse)
    return data


def preprocess_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    fout = open(filepath + '_v2', 'w')
    column_names = lines[6].split()[1:]
    fout.write(','.join(column_names))
    fout.write('\n')
    for line in lines[8:-1]:
        elements = line.split()
        fout.write(','.join(elements))
        fout.write('\n')
    fout.close()


def lists_identical(list1, list2):
    list1.sort()
    list2.sort()
    if list1 == list2:
        return True
    else:
        return False


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def netflow_encoding(flow, mappings):
    """
    The netflow encoding described in Pellegrino, Gaetano, et al. "Learning Behavioral Fingerprints From Netflows Using
    Timed Automata."
    :param flow: the flow to be given a code
    :param df: the dataframe with all flows
    :param mappings: dictionary with the features to be used for encoding and their cardinality
    :return: the code that represents the flow
    """
    code = 0
    space_size = reduce((lambda x, y: x * y), list(mappings.values()))
    for feature in mappings.keys():
        code += flow[feature + '_num'] * space_size / mappings[feature]
        space_size = space_size / mappings[feature]
    return code


def netflow_encoding_str(flow, mappings):
    """
    The netflow encoding described in Pellegrino, Gaetano, et al. "Learning Behavioral Fingerprints From Netflows Using
    Timed Automata."
    :param flow: the flow to be given a code
    :param df: the dataframe with all flows
    :param mappings: dictionary with the features to be used for encoding and their cardinality
    :return: the code that represents the flow
    """
    code = ''
    for feature in mappings.keys():
        code += str(flow[feature + '_num'])
    return code


def find_percentile(val, percentiles):
    """
    Helper function returning the relative index of placement in the percentiles
    :param val: the value to be indexed
    :param percentiles: the percentile limits
    :return: the index of val in the percentiles
    """
    ind = len(percentiles)
    for i, p in enumerate(percentiles):
        if val <= p:
            ind = i
            break
    return ind


def remove_background(df):
    """
    Helper function removing background flows from a given dataframe
    :param df: the dataframe
    :return: the no-background dataframe
    """
    df = df[df['label'] != 'Background']
    return df


def date_diff(d1, d2):
    # d1 = datetime.datetime.strptime(str(d1), '%Y-%m-%d %H:%M:%S.%f')
    # d2 = datetime.datetime.strptime(str(d2), '%Y-%m-%d %H:%M:%S.%f')
    d = d2 - d1
    d = int(d.total_seconds() * 1000)
    return d


def split_data(data):
    beinign_data = data[(data['label'] == 0) & (data['src_ip'] == ip_dict[scenario])]
    configuration_data, other_data = train_test_split(beinign_data, test_size=0.7, random_state=42)
    del_indices = configuration_data.index.tolist()
    data_to_return = data.drop(del_indices)
    configuration_data = configuration_data.sort_index()
    data_to_return = data_to_return.sort_index()
    return configuration_data, data_to_return


def read_and_process_data(dataset):
    data = read_data(dataset)

    columns_to_keep = ['ts', 'id.orig_h', 'proto', 'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts',
                       'label']
    data = data[columns_to_keep]
    data = data.reset_index(drop=True)
    data.columns = ['date', 'src_ip', 'protocol', 'duration', 'src_bytes', 'dst_bytes', 'src_packets', 'dst_packets',
                    'label']
    # parse packets and bytes as integers instead of strings
    # packet_median = data.groupby(['label'])['packets'].median()
    # data.set_index(['label'])['packets'].fillna(packet_median)
    data['src_bytes'] = data['src_bytes'].replace('-', np.nan)
    data['dst_bytes'] = data['dst_bytes'].replace('-', np.nan)
    data['src_packets'] = data['src_packets'].replace('-', np.nan)
    data['dst_packets'] = data['dst_packets'].replace('-', np.nan)
    data['duration'] = data['duration'].replace('-', np.nan)

    data['packets'] = data['src_packets'] + data['dst_packets']
    data['packets'].fillna(0, inplace=True)
    data['src_bytes'].fillna(0, inplace=True)
    data['dst_bytes'].fillna(0, inplace=True)
    data['duration'].fillna(0.001, inplace=True)

    # mal_flows = data[data['label'] == 'Malicious']
    # ben_flows = data[data['label'] == 'Benign']
    # mal_bytes_median = int(np.median([int(x) for x in mal_flows['bytes'] if not math.isnan(float(x))]))
    # ben_bytes_median = int(np.median([int(x) for x in ben_flows['bytes'] if not math.isnan(float(x))]))
    # bytes_median = {'Benign': ben_bytes_median, 'Malicious': mal_bytes_median}
    # data.insert(loc=0, column='bytes_', value=data.set_index(['label'])['bytes'].fillna(bytes_median).tolist())
    # data['bytes'] = data['bytes_']
    # data.drop(['bytes_'], axis=1, inplace=True)
    #
    # data['duration'] = data['duration'].replace('-', np.nan)
    # data['duration'] = data['duration'].astype(float)
    # mal_flows = data[data['label'] == 'Malicious']
    # ben_flows = data[data['label'] == 'Benign']
    # mal_duration_median = np.median([x for x in mal_flows['duration'] if not math.isnan(float(x))])
    # ben_duration_median = np.median([x for x in ben_flows['duration'] if not math.isnan(float(x))])
    # duration_median = {'Benign': ben_duration_median, 'Malicious': mal_duration_median}
    # # data['duration'] = data.set_index(['label'])['duration'].fillna(duration_median).reset_index()
    # data.insert(loc=0, column='duration_', value=data.set_index(['label'])['duration'].fillna(duration_median).tolist())
    # data['duration'] = data['duration_']
    # data.drop(['duration_'], axis=1, inplace=True)

    # add the numerical representation of the categorical data
    data['protocol_num'] = pd.Categorical(data['protocol'], categories=data['protocol'].unique()).codes
    data['label'] = [0 if x == 'Benign' else 1 for x in data['label']]

    data['duration'] = data['duration'].astype(float)
    data['packets'] = data['packets'].astype(int)
    data['src_bytes'] = data['src_bytes'].astype(int)
    data['dst_bytes'] = data['dst_bytes'].astype(int)
    data = data[data['date'].notna()]
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index(['date'])
    data = data.sort_index()
    return data


def extract_traces(dataset='CTU-Malware-Capture-Botnet-50', data_dir='traces_ctu_50', scenario='scenario_50',
                   window_size=800, stride_size=400):
    # normal_ips = ip_dict[scenario]['normal_ips']
    # infected_ips = ip_dict[scenario]['infected_ips']
    # ## Read the dataset for scenario 10

    if not os.path.exists('../../data/' + data_dir):
        os.makedirs('../../data/' + data_dir)
    partial_name = "/" + ("_".join(data_dir.split('/')[-1].split('_')[1:]))

    print("Starting")
    data = read_and_process_data(dataset)
    data = data[data['src_ip'] == ip_dict[scenario]]
    # src_ips = list(set(data['src_ip'].tolist()))
    # infected_ips = []
    # for i, flow in data.iterrows():
    #     if flow['label'] == 1:
    #         infected_ips += [flow['src_ip']]
    # infected_ips = list(set(srp_ips))
    # normal_ips = list(set(srp_ips) - set(infected_ips))

    # experiment
    # total = len(list(srp_ips))
    # diff_labels = 0
    # pos = 0
    # neg = 0
    # for srp_ip in list(srp_ips):
    #     temp_data = data[data['src_ip'] == srp_ip]
    #     labels = list(set(temp_data['label'].tolist()))
    #     if len(labels) > 1:
    #         diff_labels += 1
    #         print('Problema')
    #     else:
    #         if int(labels[0]) == 0:
    #             neg += 1
    #         else:
    #             pos += 1

    # print('Percentage of IPs with multiple labels: ', 100 * float(diff_labels) / total)
    # print('Percentage of IPs with positive label: ', 100 * float(pos) / total)
    # print('Percentage of IPs with negative label: ', 100 * float(neg) / total)
    # return
    # experiment
    # ## Pick one infected host and the normal ones

    # src_ips = set(data['src_ip'].tolist())
    # for src_ip in src_ips:
    #     print(len(data[data['src_ip']==src_ip]))

    # separate the types of features in the dataset
    # continuous_features = ['duration', 'protocol_num', 'flags_num', 'tos', 'packets', 'bytes', 'flows']
    continuous_features = ['duration', 'packets', 'src_bytes', 'dst_bytes']
    # categorical_features = ['protocol', 'flags', 'direction']
    categorical_features = ['protocol', 'direction']

    # ## Select the wanted features and apply clustering in case they are numerical

    # selected_features = ['duration', 'protocol', 'flags', 'packets', 'bytes']
    selected_features = ['duration', 'protocol', 'packets', 'src_bytes', 'dst_bytes']
    # configuration_data = data[(data['src_ip'] == normal_ips[0]) | (data['dst_ip'] == normal_ips[0])]
    # configuration_data = data[(data['src_ip'] == normal_ips[0])]
    # data = data.reset_index()
    configuration_data, data = split_data(data)
    data = data.sort_index()

    percentile_num = {}
    for sel in selected_features:
        if sel in continuous_features:
            # apply the elbow method
            print('----------------------- Finding optimal number of bins for {} -----------------------'.format(sel))
            Sum_of_squared_distances = []
            for k in range(1, 11):
                km = KMeans(n_clusters=k)
                km = km.fit(configuration_data[sel].values.reshape(-1, 1))
                Sum_of_squared_distances.append(km.inertia_)

            plt.figure()
            plt.plot(range(1, 11), Sum_of_squared_distances, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Sum_of_squared_distances')
            plt.title('Elbow Method For Optimal k')
            plt.grid()
            plt.show()
            #         plt.savefig('plots/elbow_discretization_%s.png' % sel, bbox_inches='tight')

            percentile_num[sel] = int(input('Enter your preferred number of clusters: '))

            # assign the cluster id to each value of the selected numerical feature in the way that it is described in

    # all_ips = []
    # for ip in set(data['src_ip'].tolist()):
    #     if len(data[data['src_ip'] == ip]) >= 10:
    #         all_ips += [ip]
    # all_ips = set(data['src_ip'].tolist())  # exclude configuration IP
    configuration_done = False
    for label in [0, 1]:
        for i in range(2):
            if i == 0:
                ip = ip_dict[scenario]
                if label == 1:
                    filename = data_dir + partial_name + '_infected_' + ip + '_traces'
                else:
                    filename = data_dir + partial_name + '_benign' + ip + '_traces'

                # selected_data = data[(data['src_ip'] == ip) | (data['dst_ip'] == ip)] #maybe that's important
                selected_data = data[(data['src_ip'] == ip) & (data['label'] == label)]
                # selected_data = data[(data['src_ip'] == infected_ip) | (data['dst_ip'] == infected_ip)]
                # selected_data = selected_data.reset_index(level=0, drop=True).reset_index()
                # if selected_data.empty:
                if len(selected_data) < 10:
                    # print("Traces for {} are empty".format(ip))
                    print("Traces for {} are less than 10".format(ip))
                    continue
            else:
                if configuration_done:
                    continue
                selected_data = configuration_data.copy()
                filename = data_dir + partial_name + '_configuration_traces'
                configuration_done = True
            # ## Discretize the flows
            # discretize all flows
            print('Discretizing all hosts...')
            for sel in selected_features:
                if sel in continuous_features:
                    # in Pellegrino, Gaetano, et al. "Learning Behavioral Fingerprints From Netflows Using Timed Automata."
                    percentile_values = list(
                        map(lambda p: np.percentile(selected_data[sel], p),
                            100 * np.arange(0, 1, 1 / percentile_num[sel])[1:]))
                    selected_data[sel + '_num'] = selected_data[sel].apply(find_percentile, args=(percentile_values,))

            mappings = {}
            for sel_feat in selected_features:
                mappings[sel_feat] = len(selected_data[sel_feat + '_num'].unique())
            # selected_data['encoded'] = selected_data.apply(lambda x: netflow_encoding(x, mappings), axis=1)
            selected_data['encoded'] = selected_data.apply(lambda x: netflow_encoding(x, mappings), axis=1)
            selected_data.to_pickle('discretized_data/all_discretized_%s.pkl' % '_'.join(selected_features))
            print('Discretization completed')

            mapping_to_integers = list(range(1, len(selected_data['encoded'].unique()) + 1))
            for i, v in enumerate(sorted(list(selected_data['encoded'].unique()))):
                selected_data['encoded'][selected_data['encoded'] == v] = mapping_to_integers[i]
            # selected_data = selected_data.set_index(['date'])

            # selected_data = selected_data.sort_index()

            # data['date'] = data['date'].apply(Timestamp)
            # data = data.sort_values(by=['date'], ascending=False)

            # selected_data.rolling(window='20ms')
            # experiments

            # experiments
            dates = selected_data.index.tolist()
            date_diffs = [date_diff(s, t) for s, t in zip(dates, dates[1:])]
            # date_diffs = [d for d in date_diffs if d != 0]
            if math.isnan(np.median(date_diffs)):
                continue
            elif np.median(date_diffs) == 0:
                date_median = int(round(np.mean(date_diffs)))
            else:
                date_median = int(np.median(date_diffs))
            print('Started extracting sliding windows')
            print('Window = {}, Stride = {}'.format(str(date_median * 100), str(date_median * 50)))
            # selected_data['encoded'].rolling(window='10ms').apply(lambda x: foo(x, windows))
            starting_date = selected_data.index.tolist()[0].strftime('%Y-%m-%d %H:%M:%S.%f')
            ending_date = selected_data.index.tolist()[-1].strftime('%Y-%m-%d %H:%M:%S.%f')
            r = pd.date_range(start=starting_date, end=ending_date, freq=str(date_median * 50) + 'ms').strftime(
                '%Y-%m-%d %H:%M:%S.%f').values
            # r = pd.date_range(start=starting_date, end=ending_date, freq='{}ms'.format(stride_size)).strftime(
            #     '%Y-%m-%d %H:%M:%S.%f').values
            if len(r) >= 3:
                windows_dates = list(
                    map(tuple, rolling_window(r, int(window_size / stride_size) + 1)[:, [0, -1]].tolist()))
            else:
                continue
            windows = []
            prev_window = []
            per = 0.1
            for i, windows_date in enumerate(windows_dates):
                if i > len(windows_dates) * per:
                    print(int(per * 100), "% completed")
                    per += 0.1
                starting_date = windows_date[0]
                ending_date = windows_date[1]
                mask = (selected_data.index > starting_date) & (selected_data.index <= ending_date)
                selected_subset = selected_data.loc[mask]
                new_window = [int(v['encoded']) for _, v in selected_subset.iterrows()]
                if not lists_identical(prev_window, new_window) and new_window:
                    windows += [new_window]
                    prev_window = new_window

            print('Finished extracting sliding windows')

            lengths = []
            zeros = 0
            for win in windows:
                lengths += [len(win)]
                if len(win) == 0:
                    zeros += 1
            if len(windows) > 0:
                print('Percentage of empty windows: ', zeros / len(windows))
            else:
                print('Percentage of empty windows: ', 0)
            print('Number of traces: ', len(windows))
            if lengths:
                print('Mean window size: ', np.mean(lengths))
                print('Std window size: ', np.std(lengths))
                print('Median window size: ', np.median(lengths))
                print('Min window size: ', np.min(lengths))
                print('Max window size: ', np.max(lengths))

            with open('../../data/' + filename, "w") as fout:
                num_of_traces = len(windows)
                num_of_symbols = len(selected_data['encoded'].unique())
                fout.write(str(num_of_traces) + " " + str(num_of_symbols) + "\n")
                for i, win in enumerate(windows):
                    win_length = len(win)
                    fout.write(str(win_length) + " ")
                    if i != len(windows) - 1:
                        for trace in win:
                            fout.write(str(trace) + " ")
                    else:
                        for trace in win:
                            fout.write(str(trace) + " ")
                    fout.write("\n")


if __name__ == '__main__':
    scenario = '34-1'
    dataset = '../../data/IoTScenarios/CTU-IoT-Malware-Capture-{}/bro/conn.log.labeled'.format(scenario)
    window_size = 600
    stride_size = 300
    # data_dir = 'ctu_13/{}/{}-{}_traces_ctu_51'.format(scenario, window_size, stride_size)
    data_dir = 'IOT/{}/flexible_traces_ctu_51'.format(scenario)
    extract_traces(dataset, data_dir, scenario, window_size, stride_size)
