#!/usr/bin/env python
# coding: utf-8
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from sklearn.cluster import KMeans

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
prev_parentdir = os.path.dirname(parentdir)
sys.path.insert(0, prev_parentdir)

from pypackage.flexfringe.utils import preprocess_bidirectional_data, rolling_window, lists_identical, netflow_encoding, \
    find_percentile, remove_background, date_diff, split_data

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


# ## Functions used


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

    # parse packets and bytes as integers instead of strings
    data['packets'] = data['packets'].astype(int)
    data['duration'] = data['duration'].astype(float)
    data['src_bytes'] = data['src_bytes'].astype(int)
    data['dst_bytes'] = data['dst_bytes'].astype(int)

    # add the numerical representation of the categorical data
    data['protocol_num'] = pd.Categorical(data['protocol'], categories=data['protocol'].unique()).codes
    # data['state_num'] = pd.Categorical(data['state'], categories=data['state'].unique()).codes
    data['direction_num'] = pd.Categorical(data['direction'], categories=data['direction'].unique()).codes

    return data


def get_precentiles(all_conf_data):
    # ## Read the dataset for scenario 10
    continuous_features = ['duration', 'packets', 'src_bytes', 'dst_bytes']
    # ## Select the wanted features and apply clustering in case they are numerical
    selected_features = ['duration', 'protocol', 'packets', 'src_bytes', 'dst_bytes']
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

            plt.figure()
            plt.plot(range(1, 11), Sum_of_squared_distances, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Sum_of_squared_distances')
            plt.title('Elbow Method For Optimal k')
            plt.grid()
            plt.show()
            # plt.savefig("test.png", bbox_inches='tight')
            percentile_num[sel] = int(input('Enter your preferred number of clusters: '))
    return percentile_num


def discretize(data, data_dir, scenario, window_size, stride_size, percentile_num, case='train', conf=False):
    data = data.set_index(['date'])
    data = data.sort_index()
    if not conf:
        infected_ips = ip_dict[scenario]['infected_ips']

    # separate the types of features in the dataset
    continuous_features = ['duration', 'packets', 'src_bytes', 'dst_bytes']
    categorical_features = ['protocol']

    # ## Select the wanted features and apply clustering in case they are numerical

    selected_features = ['duration', 'protocol', 'packets', 'src_bytes', 'dst_bytes']
    partial_name = "/" + ("_".join(data_dir.split('/')[-1].split('_')[1:]))

    if case == 'train':
        all_connections = [group for _, group in data.groupby(['src_ip', 'dst_ip']) if len(group) > 500]  # change
    else:
        all_connections = [group for _, group in data.groupby(['src_ip', 'dst_ip'])]

    for selected_data in all_connections:
        src_ip = selected_data.iloc[0]['src_ip']
        dst_ip = selected_data.iloc[0]['dst_ip']
        ip = src_ip + '-' + dst_ip  # if ip not in set(data['src_ip'].tolist()):
        #     continue
        if not conf:
            if src_ip in infected_ips:
                filename = data_dir + '/infected_' + ip + '_traces'
            else:
                filename = data_dir + '/benign' + ip + '_traces'
        else:
            filename = data_dir + '/configuration_' + ip + '_traces'

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
        # selected_data.to_pickle('discretized_data/all_discretized_%s.pkl' % '_'.join(selected_features))
        print('Discretization completed')

        mapping_to_integers = list(range(1, len(selected_data['encoded'].unique()) + 1))
        for i, v in enumerate(sorted(list(selected_data['encoded'].unique()))):
            selected_data['encoded'][selected_data['encoded'] == v] = mapping_to_integers[i]
        # experiments

        print('Started extracting sliding windows')

        # print('Window = {}, Stride = {}'.format('20ms', '1 flow'))
        # windows = []
        # selected_data['encoded'].rolling(window='20ms').apply(lambda x: foo(x, windows))
        selected_data = selected_data['encoded']

        # windows with manual way
        dates = selected_data.index.tolist()
        if len(dates) < 2:
            continue
        date_diffs = [date_diff(s, t) for s, t in zip(dates, dates[1:])]
        # date_diffs = [d for d in date_diffs if d != 0]
        date_median = int(np.median(date_diffs))
        if date_median == 0:
            date_median += 1
        print('Window = {}, Stride = {}'.format(str(date_median * window_size), str(date_median * window_size / 2)))
        starting_date = selected_data.index.tolist()[0].strftime('%Y/%m/%d %H:%M:%S.%f')
        ending_date = selected_data.index.tolist()[-1].strftime('%Y/%m/%d %H:%M:%S.%f')
        r = pd.date_range(start=starting_date, end=ending_date,
                          freq=str(date_median * window_size / 2) + 'ms').strftime(
            '%Y-%m-%d %H:%M:%S.%f').values
        # r = pd.date_range(start=starting_date, end=ending_date, freq='{}ms'.format(stride_size)).strftime(
        #     '%Y-%m-%d %H:%M:%S.%f').values
        if len(r) >= 3:
            windows_dates = list(map(tuple, rolling_window(r, int(window_size / stride_size) + 1)[:, [0, -1]].tolist()))
        else:
            windows_dates = [(starting_date, ending_date)]
        windows = []
        prev_window = []
        per = 0.1
        for i, windows_date in enumerate(windows_dates):
            if i > len(windows_dates) * per:
                print(int(per * 100), "% completed")
                per += 0.1
            # selected_series = selected_data['encoded']
            mask = (selected_data.index > windows_date[0]) & (selected_data.index <= windows_date[1])
            data_subset = selected_data.loc[mask]
            new_window = [int(v) for v in data_subset.tolist()]
            if not lists_identical(prev_window, new_window) and new_window:
                windows += [new_window]
                prev_window = new_window
        # end windows with manual way
        print('Finished extracting sliding windows')
        if windows:
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

            with open('../data/' + filename, "w") as fout:
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
    # ## Read the dataset for scenario 10

    if not os.path.exists('../data/' + data_dir):
        os.makedirs('../data/' + data_dir)

    if multiple:
        data = data_dict[int(scenario.split('_')[-1])]
    else:
        data = read_and_process_data(dataset)
    print("Starting")
    # ## Pick one infected host and the normal ones

    if not multiple:
        configuration_data, data = split_data(data)
        configuration_data = configuration_data.sort_index()
        data = data.sort_index()
        percentile_num = get_precentiles(configuration_data)
        discretize(configuration_data.copy(), scenario, data_dir, window_size, stride_size, percentile_num,
                   case=case, conf=True)

    discretize(data.copy(), data_dir, scenario, window_size, stride_size, percentile_num, case=case, conf=False)


def extract_conf_traces(window_size, stride_size, all_conf_data, percentile_num, conf_data_dir, scenario):
    discretize(all_conf_data.copy(), conf_data_dir, scenario, window_size, stride_size, percentile_num,
               case='train', conf=True)


def discretize_for_single_scenario():
    window_size = 1000
    stride_size = 500
    scenarios = list(range(42, 55))
    training_sets = [44, 45, 46, 48, 51, 52, 52, 54]
    for s in scenarios:
        scenario = 'scenario_{}'.format(s)
        dataset = 'CTU-Malware-Capture-Botnet-{}'.format(s)
        data_dir = 'discretized_data/ctu_13/connection/single_scenario/{}/{}-{}_traces_ctu_{}'.format(scenario,
                                                                                                              window_size,
                                                                                                              stride_size,
                                                                                                              s)
        if s in training_sets:
            case = 'train'
        else:
            case = 'test'
        extract_traces(dataset, data_dir, scenario, window_size,
                       stride_size, {}, {}, case=case, multiple=False)


def discretize_for_multiple_scenarios():
    window_size = 500
    stride_size = 250
    scenarios = list(range(42, 55))
    training_sets = [44, 45, 46, 48, 51, 52, 52, 54]
    all_conf_data = pd.DataFrame()
    data_dict = {}
    for s in scenarios:
        dataset = 'CTU-Malware-Capture-Botnet-{}'.format(s)
        if s in training_sets:
            all_data = read_and_process_data(dataset)
            conf_data, data = split_data(all_data)
            all_conf_data = pd.concat([all_conf_data, conf_data])
            data_dict[s] = data.sort_index()
        else:
            data_dict[s] = read_and_process_data(dataset)

    all_conf_data = all_conf_data.sort_index()
    # # percentile_num = get_precentiles(all_conf_data)
    percentile_num = {'duration': 2, 'packets': 3, 'src_bytes': 2, 'dst_bytes': 2}
    conf_dir = 'discretized_data_{}median_{}median/ctu_13/connection/multiple_scenarios/configuration_data/'.format(window_size,
                                                                                                        stride_size)
    extract_conf_traces(window_size, stride_size, all_conf_data, percentile_num, conf_dir, '')
    for s in scenarios:
        scenario = 'scenario_{}'.format(s)
        dataset = 'CTU-Malware-Capture-Botnet-{}'.format(s)
        data_dir = 'discretized_data_{}median_{}median/ctu_13/connection/multiple_scenarios/{}/{}-{}_traces_ctu_{}'.format(
            window_size,
            stride_size,
            scenario,
            window_size,
            stride_size,
            s)

        if s in training_sets:
            extract_traces(dataset, data_dir, scenario, window_size, stride_size, data_dict, percentile_num,
                           'train', multiple=True)
        else:
            extract_traces(dataset, data_dir, scenario, window_size, stride_size, data_dict, percentile_num,
                           'test', multiple=True)


if __name__ == '__main__':
    # discretize_for_single_scenario()
    discretize_for_multiple_scenarios()
