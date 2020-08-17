#!/usr/bin/env python
# coding: utf-8
import inspect
import math
import os
import sys
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.cluster import KMeans

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from pypackage.flexfringe.utils import preprocess_bidirectional_data, rolling_window, lists_identical, netflow_encoding, \
    find_percentile, date_diff, split_data, remove_background, foo, split_data2, split_data_multiple

# warnings.filterwarnings("ignore")
ip_dict = {}

# IPs for scenario 42
infected_ips = ['147.32.84.165']
normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

ip_dict['scenario_42'] = {}
ip_dict['scenario_42']['infected_ips'] = infected_ips
ip_dict['scenario_42']['normal_ips'] = normal_ips

# IPs for scenario 43
infected_ips = ['147.32.84.165']
normal_ips = ['147.32.84.170', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

ip_dict['scenario_43'] = {}
ip_dict['scenario_43']['infected_ips'] = infected_ips
ip_dict['scenario_43']['normal_ips'] = normal_ips

# IPs for scenario 44
infected_ips = ['147.32.84.165']
normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

ip_dict['scenario_44'] = {}
ip_dict['scenario_44']['infected_ips'] = infected_ips
ip_dict['scenario_44']['normal_ips'] = normal_ips

# IPs for scenario 45
infected_ips = ['147.32.84.165']
normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

ip_dict['scenario_45'] = {}
ip_dict['scenario_45']['infected_ips'] = infected_ips
ip_dict['scenario_45']['normal_ips'] = normal_ips

# IPs for scenario 46
infected_ips = ['147.32.84.165']
normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

ip_dict['scenario_46'] = {}
ip_dict['scenario_46']['infected_ips'] = infected_ips
ip_dict['scenario_46']['normal_ips'] = normal_ips

# IPs for scenario 47
infected_ips = ['147.32.84.165']
normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

ip_dict['scenario_47'] = {}
ip_dict['scenario_47']['infected_ips'] = infected_ips
ip_dict['scenario_47']['normal_ips'] = normal_ips

# IPs for scenario 48
infected_ips = ['147.32.84.165']
normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9']

ip_dict['scenario_48'] = {}
ip_dict['scenario_48']['infected_ips'] = infected_ips
ip_dict['scenario_48']['normal_ips'] = normal_ips

# IPs for scenario 49
infected_ips = ['147.32.84.165']
normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

ip_dict['scenario_49'] = {}
ip_dict['scenario_49']['infected_ips'] = infected_ips
ip_dict['scenario_49']['normal_ips'] = normal_ips

# IPs for scenario 50
infected_ips = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204',
                '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']
normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

ip_dict['scenario_50'] = {}
ip_dict['scenario_50']['infected_ips'] = infected_ips
ip_dict['scenario_50']['normal_ips'] = normal_ips

# IPs for scenario 51
normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']
infected_ips = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204',
                '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']

ip_dict['scenario_51'] = {}
ip_dict['scenario_51']['infected_ips'] = infected_ips
ip_dict['scenario_51']['normal_ips'] = normal_ips

# IPs for scenario 52
infected_ips = ['147.32.84.165', '147.32.84.191', '147.32.84.192']
normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

ip_dict['scenario_52'] = {}
ip_dict['scenario_52']['infected_ips'] = infected_ips
ip_dict['scenario_52']['normal_ips'] = normal_ips

# IPs for scenario 53
infected_ips = ['147.32.84.165', '147.32.84.191', '147.32.84.192']
normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

ip_dict['scenario_53'] = {}
ip_dict['scenario_53']['infected_ips'] = infected_ips
ip_dict['scenario_53']['normal_ips'] = normal_ips

# IPs for scenario 54
infected_ips = ['147.32.84.165']
normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

ip_dict['scenario_54'] = {}
ip_dict['scenario_54']['infected_ips'] = infected_ips
ip_dict['scenario_54']['normal_ips'] = normal_ips


selected_features = ['duration', 'protocol', 'total_bytes', 'src_bytes', 'time_diff', 'packets']
continuous_features = ['duration', 'packets', 'src_bytes', 'dst_bytes', 'total_bytes', 'time_diff']
feature_dir = '_'.join(selected_features)



# Reads the dataset and returns it in DataFrame format
def read_and_process_data(dataset):
    dataset_dir = '../data/ctu/'
    if os.path.exists(dataset_dir + 'no_background_' + dataset + '.pkl'):
        data = pd.read_pickle(
            dataset_dir + 'no_background_' + dataset + '.pkl')  # if the data without the background are there, load them

    else:
        # read the data in chunks due to their large size - uncomment the following lines if you want to read them again
        # and store them in a pickle
        preprocess_bidirectional_data(dataset_dir + dataset)
        dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d %H:%M:%S.%f')
        data = pd.concat(
            remove_background(chunk) for chunk in pd.read_csv(dataset_dir + dataset + '_v2',
                                                              chunksize=100000, delimiter=',',
                                                              parse_dates=['date'], date_parser=dateparse))
        data.to_pickle(dataset_dir + 'no_background_' + dataset + '.pkl')

        # ## Initial preprocessing of the data
        # resetting indices for data
    data = data.reset_index(drop=True)
    dates = data['date'].tolist()
    time_diff = [int(abs(date_diff(s, t))) for s, t in zip(dates[1:], dates)]
    time_diff = [np.mean(time_diff)] + time_diff
    # parse packets and bytes as integers instead of strings
    data['time_diff'] = time_diff
    # parse packets and bytes as integers instead of strings
    data['packets'] = data['packets'].astype(int)
    data['duration'] = data['duration'].astype(float)
    data['src_bytes'] = data['src_bytes'].astype(int)
    data['dst_bytes'] = data['dst_bytes'].astype(int)
    data['total_bytes'] = data['total_bytes'].astype(int)

    # add the numerical representation of the categorical data
    data['protocol_num'] = pd.Categorical(data['protocol'], categories=data['protocol'].unique()).codes
    # data['state_num'] = pd.Categorical(data['state'], categories=data['state'].unique()).codes
    data['direction_num'] = pd.Categorical(data['direction'], categories=data['direction'].unique()).codes

    return data


# Computes the percentiles needed for the discretization of the flows
def get_precentiles(conf_data_dict):
    if isinstance(conf_data_dict, dict):
        all_conf_data = pd.DataFrame()

        for k, v in conf_data_dict.items():
            all_conf_data = pd.concat([all_conf_data, v])
    else:
        all_conf_data = conf_data_dict

    configuration_data = all_conf_data
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
            distances = []
            for i in range(10):
                p1 = np.asarray([1, Sum_of_squared_distances[0]])
                p2 = np.asarray([10, Sum_of_squared_distances[9]])
                p = np.asarray([i + 1, Sum_of_squared_distances[i]])
                d = norm(np.cross(p2 - p1, p1 - p)) / norm(p2 - p1)
                distances.append(d)
            index, element = max(enumerate(distances), key=itemgetter(1))
            percentile_num[sel] = index + 1
            # plt.figure()
            # plt.plot(range(1, 11), Sum_of_squared_distances, 'bx-')
            # plt.xlabel('k')
            # plt.ylabel('Sum_of_squared_distances')
            # plt.title('Elbow Method For Optimal k')
            # plt.grid()
            # plt.show()
            # # plt.savefig("test.png", bbox_inches='tight')
            # percentile_num[sel] = int(input('Enter your preferred number of clusters: '))
    return percentile_num


def encode(data, data_dir, scenario, window_size, stride_size, percentile_num, case='train', conf=False):
    data = data.set_index(['date'])
    data = data.sort_index()
    if not conf:
        infected_ips = ip_dict[scenario]['infected_ips']

    if case == 'train':
        all_hosts = [group for _, group in data.groupby(['src_ip']) if len(group) > 500]  # change
    else:
        all_hosts = [group for _, group in data.groupby(['src_ip'])]

    for selected_data in all_hosts:
        src_ip = selected_data.iloc[0]['src_ip']
        if not conf:
            if src_ip in infected_ips:
                filename = data_dir + '/infected_' + src_ip + '_traces_' + str(len(selected_data))
            else:
                filename = data_dir + '/benign' + src_ip + '_traces_' + str(len(selected_data))
        else:
            filename = data_dir + '/configuration_' + src_ip + '_traces_' + str(len(selected_data))
        filename = filename.replace('//', '/')
        if not os.path.exists('../data/' + '/'.join(filename.split('/')[:-1])):
            os.makedirs('../data/' + '/'.join(filename.split('/')[:-1]))
        selected_data = selected_data.sort_index()
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
        selected_data['encoded'] = selected_data.apply(lambda x: netflow_encoding(x, mappings), axis=1)
        print('Discretization completed')

        mapping_to_integers = list(range(1, len(selected_data['encoded'].unique()) + 1))
        for i, v in enumerate(sorted(list(selected_data['encoded'].unique()))):
            selected_data['encoded'][selected_data['encoded'] == v] = mapping_to_integers[i]

        print('Started extracting sliding windows')

        windows = []
        selected_data = selected_data['encoded']
        if len(selected_data) < window_size:
            window_size = len(selected_data)
            stride_size = math.ceil(window_size / 2)
        selected_data.rolling(window=window_size).apply(lambda x: foo(x, windows))
        windows = windows[::stride_size]

        print('Finished extracting sliding windows')
        if windows:
            lengths = []
            for win in windows:
                lengths += [len(win)]
            with open('../data/' + filename + '.txt', "w") as fout:
                num_of_traces = len(windows)
                num_of_symbols = len(selected_data.unique())
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


def extract_traces(dataset, data_dir, scenario,
                   window_size, stride_size, data_dict, percentile_num, case='train', multiple=False):
    if not os.path.exists('../data/' + data_dir):
        os.makedirs('../data/' + data_dir)

    if multiple:
        data = data_dict[int(scenario.split('_')[-1])].copy()
    else:
        data = dataset
    print("Starting")

    encode(data.copy(), data_dir, scenario, window_size, stride_size, percentile_num, case=case, conf=False)


def extract_conf_traces(window_size, stride_size, conf_data_dict, percentile_num, conf_data_dir, scenario,
                        case='train'):
    if isinstance(conf_data_dict, dict):
        all_conf_data = pd.DataFrame()

        for k, v in conf_data_dict.items():
            all_conf_data = pd.concat([all_conf_data, v])

    else:
        all_conf_data = conf_data_dict

    all_conf_data = all_conf_data.sort_index()
    encode(all_conf_data.copy(), conf_data_dir, scenario, window_size, stride_size,
           percentile_num,
           case=case, conf=True)


def encode_for_single_scenario():
    window_size = 20
    stride_size = 10
    scenarios = [50, 51, 52, 53]
    for s in scenarios:
        scenario = 'scenario_{}'.format(s)
        dataset = 'CTU-Malware-Capture-Botnet-{}'.format(s)
        # data_dir = 'discretized_fixed20_10_final/ctu13_host/single/{}/{}/'.format(feature_dir, scenario)
        data_dir = 'test_1_2/ctu13_host/single/{}/{}/'.format(feature_dir, scenario)

        normal_ips = ip_dict[scenario]['normal_ips']

        data = read_and_process_data(dataset)
        configuration_data, data = split_data2(data, normal_ips)
        configuration_data = configuration_data.sort_index()
        data = data.sort_index()
        percentile_num = get_precentiles(configuration_data)
        extract_conf_traces(window_size, stride_size, configuration_data.copy(), percentile_num, data_dir,
                                     scenario, case='test')
        extract_traces(data, data_dir, scenario, window_size,
                       stride_size, {}, percentile_num, case='test', multiple=False)


def encode_for_multiple_scenarios():
    window_size = 20
    stride_size = 10
    scenarios = list(range(42, 55))
    training_sets = [44, 45, 46, 48, 51, 52, 52, 54]
    data_dict = {}
    conf_data_dict = {}
    for s in scenarios:
        dataset = 'CTU-Malware-Capture-Botnet-{}'.format(s)
        if s in training_sets:
            all_data = read_and_process_data(dataset)
            conf_data, data = split_data2(all_data)
            # conf_data, data = split_data_multiple(all_data)
            conf_data_dict[s] = conf_data.sort_index()
            data_dict[s] = data.sort_index()
        else:
            data_dict[s] = read_and_process_data(dataset)

    percentile_num = get_precentiles(conf_data_dict)
    conf_dir = 'discretized_fixed20_10_final/ctu13_host/multi/{}/configuration_data/'.format(feature_dir)
    extract_conf_traces(window_size, stride_size, conf_data_dict, percentile_num, conf_dir, '', case='test')
    for s in scenarios:
        scenario = 'scenario_{}'.format(s)
        dataset = 'CTU-Malware-Capture-Botnet-{}'.format(s)
        data_dir = 'discretized_fixed20_10_final/ctu13_host/multi/{}/{}/'.format(feature_dir, scenario)

        if s in training_sets:
            extract_traces(dataset, data_dir, scenario, window_size, stride_size, data_dict,
                           percentile_num,
                           'train', multiple=True)
        else:
            extract_traces(dataset, data_dir, scenario, window_size, stride_size, data_dict,
                           percentile_num,
                           'test', multiple=True)


if __name__ == '__main__':
    encode_for_single_scenario()
    # encode_for_multiple_scenarios()
