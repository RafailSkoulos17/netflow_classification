#!/usr/bin/env python
# coding: utf-8
import inspect
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from pypackage.flexfringe.utils import rolling_window, lists_identical, netflow_encoding, find_percentile, \
    date_diff, split_data
import datetime


# ## Functions used


def read_and_process_data(dataset):
    filename = '/'.join(dataset.split('/')[:-1]) + '/' + dataset.split('/')[-1].rstrip('.csv')
    if os.path.exists(filename + '.pickle'):
        data = pd.read_pickle(filename + '.pickle')
    else:
        try:
            dateparse = lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

            data = pd.concat(
                chunk for chunk in pd.read_csv(dataset, delimiter=',',
                                               engine='python', parse_dates=[' Timestamp'],
                                               date_parser=dateparse,
                                               chunksize=10000))
        except ValueError:
            dateparse = lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H:%M')

            data = pd.concat(
                chunk for chunk in pd.read_csv(dataset, delimiter=',',
                                               engine='python', parse_dates=[' Timestamp'],
                                               date_parser=dateparse,
                                               chunksize=10000))
        data.to_pickle(filename + '.pickle')

        # ## Initial preprocessing of the data
        # resetting indices for data
    data = data.reset_index(drop=True)

    data.columns = [d.lstrip() for d in data.columns]
    data['src_ip'] = data['Source IP']
    data['dst_ip'] = data['Destination IP']
    data['protocol'] = data['Protocol']
    data['date'] = data['Timestamp']
    data['duration'] = data['Flow Duration']
    data['packets'] = data['Total Fwd Packets'] + data['Total Backward Packets']
    data['label'] = data['Label']
    data['src_bytes'] = data['Subflow Fwd Bytes']
    data['dst_bytes'] = data['Subflow Bwd Bytes']

    columns_to_keep = ['date', 'src_ip', 'dst_ip', 'protocol', 'duration', 'src_bytes', 'dst_bytes', 'packets', 'label']
    data = data[columns_to_keep]
    data = data.reset_index(drop=True)
    data = data.dropna()

    data['packets'] = data['packets'].astype(int)
    data['duration'] = data['duration'].astype(float)
    data['src_bytes'] = data['src_bytes'].astype(int)
    data['dst_bytes'] = data['dst_bytes'].astype(int)

    # add the numerical representation of the categorical data
    data['protocol_num'] = pd.Categorical(data['protocol'], categories=data['protocol'].unique()).codes
    data['label'] = [0 if d == 'BENIGN' else 1 for d in data['label']]

    print(list(set(data['label'].tolist())))
    print('Malicious: ', str(len(data[data['label'] == 1])))
    print('Benign: ', str(len(data[data['label'] == 0])))
    print("DATA READ")

    # data = data.set_index(['date'])
    # data = data.sort_index()
    return data


def discretize(data, percentile_num, data_dir, window_size, stride_size, conf=False):
    # separate the types of features in the dataset
    continuous_features = ['duration', 'packets', 'src_bytes', 'dst_bytes']
    categorical_features = ['protocol']

    # ## Select the wanted features and apply clustering in case they are numerical
    data = data.set_index(['date'])
    data = data.sort_index()
    selected_features = ['duration', 'protocol', 'packets', 'src_bytes', 'dst_bytes']
    if conf:
    	all_connections = [group for _, group in data.groupby(['src_ip', 'dst_ip']) if len(group) > 500]
    else:
	    all_connections = [group for _, group in data.groupby(['src_ip', 'dst_ip'])]

    for selected_data in all_connections:
        src_ip = selected_data.iloc[0]['src_ip']
        dst_ip = selected_data.iloc[0]['dst_ip']
        ip = src_ip + '-' + dst_ip

        if not conf:
            labels = selected_data['label'].tolist()
            most_common_label = max(set(labels), key=labels.count)
            if most_common_label == 0:
                filename = data_dir + '/benign' + ip + '_traces'
            else:
                filename = data_dir + '/infected_' + ip + '_traces'
        else:
            filename = data_dir + '/configuration_data/configuration_' + ip + '_traces'
            # filename = 'configuration_data/' + data_dir.split('/')[-1] + '/ids_configuration_traces' + ip + '_traces'
        if not os.path.exists('../data/' + '/'.join(filename.split('/')[:-1])):
            os.makedirs('../data/' + '/'.join(filename.split('/')[:-1]))
        selected_datinfected_ = selected_data.sort_index()
        # if len(selected_data) < 1000:
        #     print("Traces for {} are less than 1000".format(ip))
        #     continue
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
        # selected_data.to_pickle('discretized_data/all_discretized_%s.pkl' % '_'.join(selected_features))
        print('Discretization completed')

        mapping_to_integers = list(range(1, len(selected_data['encoded'].unique()) + 1))
        for i, v in enumerate(sorted(list(selected_data['encoded'].unique()))):
            selected_data['encoded'][selected_data['encoded'] == v] = mapping_to_integers[i]
        # selected_data = selected_data.set_index(['date'])

        selected_data = selected_data.sort_index()

        # data['date'] = data['date'].apply(Timestamp)
        # data = data.sort_values(by=['date'], ascending=False)

        # selected_data.rolling(window='20ms')
        # experiments

        # experiments

        print('Started extracting sliding windows')

        # print('Window = {}, Stride = {}'.format('20ms', '1 flow'))
        # windows = []
        # selected_data['encoded'].rolling(window='20ms').apply(lambda x: foo(x, windows))

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
        starting_date = selected_data.index.tolist()[0].strftime('%d/%m/%Y %H:%M:%S')
        ending_date = selected_data.index.tolist()[-1].strftime('%d/%m/%Y %H:%M:%S')
        r = pd.date_range(start=starting_date, end=ending_date,
                          freq=str(int(date_median * window_size / 2)) + 's').strftime(
            '%d/%m/%Y %H:%M:%S').values
        # r = pd.date_range(start=starting_date, end=ending_date, freq='{}ms'.format(stride_size)).strftime(
        #     '%d/%m/%Y %H:%M:%S:%S.%f').values
        if len(r) >= 3:
            windows_dates = list(map(tuple, rolling_window(r, int(window_size / stride_size) + 1)[:, [0, -1]].tolist()))
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


def extract_traces(dataset, data_dir, window_size, stride_size, percentile_num, multiple=False):
    if not os.path.exists('../data/' + data_dir):
        os.makedirs('../data/' + data_dir)
    print("Starting")
    data = read_and_process_data(dataset)
    data_dir = data_dir + '/' + dataset.split('/')[-1].rstrip('.csv')
    if not multiple:
        configuration_data, data = split_data(data)
        data = data.sort_index()
        configuration_data = configuration_data.sort_index()
        percentile_num = get_precentiles(configuration_data)
        discretize(configuration_data.copy(), percentile_num, data_dir, window_size, stride_size, conf=True)

    # assign the cluster id to each value of the selected numerical feature in the way that it is described in
    discretize(data.copy(), percentile_num, data_dir, window_size, stride_size, conf=False)


def extract_conf_traces(window_size, stride_size, all_conf_data, percentile_num, conf_data_dir, dataset):
    # conf_data_dir = conf_data_dir
    discretize(all_conf_data.copy(), percentile_num, conf_data_dir, window_size, stride_size, conf=True)


def get_precentiles(all_conf_data):
    # ## Read the dataset for scenario 10
    continuous_features = ['duration', 'packets', 'src_bytes', 'dst_bytes']
    # ## Select the wanted features and apply clustering in case they are numerical
    selected_features = ['duration', 'protocol', 'packets', 'src_bytes', 'dst_bytes']
    percentile_num = {}
    for sel in selected_features:
        if sel in continuous_features:
            # apply the elbow method
            print('----------------------- Finding optimal number of bins for {} -----------------------'.format(sel))
            Sum_of_squared_distances = []
            for k in range(1, 11):
                km = KMeans(n_clusters=k)
                km = km.fit(all_conf_data[sel].values.reshape(-1, 1))
                Sum_of_squared_distances.append(km.inertia_)

            plt.figure()
            plt.plot(range(1, 11), Sum_of_squared_distances, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Sum_of_squared_distances')
            plt.title('Elbow Method For Optimal k')
            plt.grid()
            # plt.show()
            plt.savefig("test.png", bbox_inches='tight')
            percentile_num[sel] = int(input('Enter your preferred number of clusters: '))
    return percentile_num


def discretize_single_scenario():
    window_size = 100
    stride_size = 50
    rootdir = '../data/IDS2017/'

    benign_scenario = 'Monday-WorkingHours.pcap_ISCX.csv'
    for subdir, dirs, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith('.pickle'):
                continue
            if filename == benign_scenario:
                continue
            else:
                # data_dir = 'discretized_data/ids/connection/single_scenario/{}'.format(filename.rstrip('.csv'))
                data_dir = 'discretized_data_100_50/ids/connection/single_scenario'
                file_path = subdir + filename
                extract_traces(file_path, data_dir, window_size, stride_size, {}, multiple=False)


def discretize_multiple_scenarios():
    window_size = 100
    stride_size = 50
    rootdir = '../data/IDS2017/'

    conf_file = 'Monday-WorkingHours.pcap_ISCX.csv'
    conf_data = read_and_process_data(rootdir + conf_file)
    percentile_num = get_precentiles(conf_data)
    # percentile_num = {'duration': 2, 'packets': 2, 'src_bytes': 4, 'dst_bytes': 2}
    conf_dir = 'discretized_data_100_50/ids/connection/multiple_scenarios'
    extract_conf_traces(window_size, stride_size, conf_data, percentile_num, conf_dir, conf_file)
    for subdir, dirs, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith('.pickle'):
                continue
            if filename == 'Monday-WorkingHours.pcap_ISCX.csv':
                continue
            else:
                # data_dir = 'discretized_data/ids/connection/multiple_scenarios/{}'.format(filename.rstrip('.csv'))
                data_dir = 'discretized_data_100_50/ids/connection/multiple_scenarios'
                file_path = subdir + filename
                extract_traces(file_path, data_dir, window_size, stride_size, percentile_num, multiple=True)


if __name__ == '__main__':
    discretize_single_scenario()
    # discretize_multiple_scenarios()
