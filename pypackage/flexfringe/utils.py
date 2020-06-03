from sklearn.model_selection import train_test_split
import os
from functools import reduce

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def read_data(filepath):
    preprocess_data(filepath)
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    data = pd.read_csv(filepath + '_v2', delimiter=',', parse_dates=['date'], date_parser=dateparse)
    return data


def preprocess_data(filepath):
    if os.path.exists(filepath + '_v2'):
        os.remove(filepath + '_v2')
    with open(filepath, 'r') as f:
        lines = f.readlines()
    fout = open(filepath + '_v2', 'w')
    column_names = ['date', 'duration', 'protocol', 'src_ip', 'src_port', 'direction', 'dst_ip', 'dst_port', 'flags',
                    'tos', 'packets', 'bytes', 'flows', 'label']
    fout.write(','.join(column_names))
    fout.write('\n')
    for line in lines[1:]:
        try:
            elements = []
            columns = [x for x in line.split('\t') if x != '']
            elements += [columns[0]]
            elements += [columns[1]]
            elements += [columns[2]]
            elements += [columns[3].split(':')[0]]
            elements += ['na' if len(columns[3].split(':')) == 1 else columns[3].split(':')[1]]
            elements += [columns[4]]
            elements += [columns[5].split(':')[0]]
            elements += ['na' if len(columns[5].split(':')) == 1 else columns[5].split(':')[1]]
            elements += [columns[6]]
            elements += [columns[7]]
            elements += [columns[8]]
            elements += [columns[9]]
            elements += [columns[10]]
            elements += [columns[11]]
            fout.write(','.join(elements))
            fout.write('\n')
        except IndexError:
            print("Error in line: ", line)
    fout.close()


def preprocess_bidirectional_data(filepath):
    if os.path.exists(filepath + '_v2'):
        os.remove(filepath + '_v2')
    with open(filepath, 'r') as f:
        lines = f.readlines()
    fout = open(filepath + '_v2', 'w')
    column_names = ['date', 'duration', 'protocol', 'src_ip', 'src_port', 'direction', 'dst_ip', 'dst_port', 'state',
                    'stos', 'dtos', 'packets', 'src_bytes', 'dst_bytes', 'total_bytes', 'label']
    fout.write(','.join(column_names))
    fout.write('\n')
    for line in lines[1:]:
        try:
            elements = []
            columns = line.split(',')
            elements += [columns[0]]  # date
            elements += [columns[1]]  # duration
            elements += [columns[2]]  # protocol
            elements += [columns[3]]  # src_ip
            elements += [columns[4]]  # src port
            elements += [columns[5]]  # direction
            elements += [columns[6]]  # dst ip
            elements += [columns[7]]  # dst port
            elements += [columns[8]]  # state
            elements += [columns[9]]  # stos
            elements += [columns[10]]  # dtos
            elements += [columns[11]]  # packets
            elements += [columns[13]]  # src bytes
            elements += [str(int(columns[12]) - int(columns[13]))]  # dst bytes
            elements += [columns[12]]  # total bytes
            elements += ['Botnet' if 'Botnet' in columns[14] else (
                'Normal' if 'Normal' in columns[14] else 'Background')]  # label
            fout.write(','.join(elements))
            fout.write('\n')
        except IndexError:
            print("Error in line: ", line)
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


def foo(x, windows):
    #     print(x)
    #     return x.sum()
    windows += [[int(v) for v in x]]
    return x.sum()


def make_barplot(data, feature):
    """
    Function for visualising the difference between categorical features for infected and normal hosts
    :param data: the dataframe containing the data
    :return:creates the wanted plot creates the wanted plot
    """
    plt.figure()
    feature_counts = (data.groupby(['is_infected'])[feature].value_counts(normalize=True).rename('percentage').mul(100)
                      .reset_index().sort_values(feature))
    ax = sns.barplot(x=feature, y='percentage', data=feature_counts, hue='is_infected',
                     palette={0: mcolors.TABLEAU_COLORS['tab:blue'], 1: mcolors.TABLEAU_COLORS['tab:red']})
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ['Normal Hosts', 'Infected host'], loc='upper right')
    plt.xlabel("%s type" % feature)
    plt.ylabel("Percentage of occurrences")
    plt.grid()
    plt.show()


#     plt.savefig('plots/barplot_%s.png' % feature)


def make_bar_graphs(x, y, feature):
    """
    Function for visualising the difference between numerical features (mainly packtes and bytes) for infected and
    normal hosts
    :param x: the type of the hosts
    :param y: the numerical values
    :param feature: the type of the feature
    :return: creates the wanted plot
    """
    plt.figure()
    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center', alpha=0.5, color=[mcolors.TABLEAU_COLORS['tab:blue'],
                                                        mcolors.TABLEAU_COLORS['tab:red']])
    plt.xticks(y_pos, x)
    plt.xlabel('Type of host')
    plt.ylabel(feature)
    plt.title('Average number of %s sent' % feature)
    plt.show()


#     plt.savefig('plots/barplot_%s.png' % feature)


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
    # d1 = datetime.datetime.strptime(str(d1), '%d/%m/%Y %H:%M:%S:%S.%f')
    # d2 = datetime.datetime.strptime(str(d2), '%d/%m/%Y %H:%M:%S:%S.%f')
    d = d2 - d1
    d = int(d.total_seconds() * 1000)
    return d


def split_data(data, normal_ips=[]):
    # beinign_data = data[(data['label'] == 0) | (data['label'] == 'Normal')]
    benign_data = data[data['src_ip'].isin(normal_ips)]
    configuration_data, other_data = train_test_split(benign_data, test_size=0.7, random_state=42)
    del_indices = configuration_data.index.tolist()
    data_to_return = data.drop(del_indices)
    configuration_data = configuration_data.sort_index()
    data_to_return = data_to_return.sort_index()
    return configuration_data, data_to_return
