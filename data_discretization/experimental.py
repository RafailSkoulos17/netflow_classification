#!/usr/bin/env python
# coding: utf-8
import inspect
import os
import sys

import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from pypackage.flexfringe.utils import remove_background, preprocess_bidirectional_data

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

    # if os.path.exists(dataset_dir + 'all_IPs_no_background_' + dataset + '.pkl'):
    #     data = pd.read_pickle(
    #         dataset_dir + 'all_IPs_no_background_' + dataset + '.pkl')  # if the data without the background are there, load them
    #
    else:
        # read the data in chunks due to their large size - uncomment the following lines if you want to read them again
        # and store them in a pickle
        preprocess_bidirectional_data(dataset_dir + dataset)
        # preprocess_data(dataset_dir + dataset)
        # dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d %H:%M:%S.%f')
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
        data = pd.concat(
            remove_background(chunk) for chunk in pd.read_csv(dataset_dir + dataset + '_v2',
                                                              chunksize=100000, delimiter=',',
                                                              parse_dates=['date'], date_parser=dateparse))
    # data = pd.concat(
    #     chunk for chunk in pd.read_csv(dataset_dir + dataset + '_v2',
    #                                    chunksize=100000, delimiter=',',
    #                                    parse_dates=['date'], date_parser=dateparse))

    # data.to_pickle(dataset_dir + 'no_background_' + dataset + '.pkl')

    # ## Initial preprocessing of the data
    # resetting indices for data
    data = data.reset_index(drop=True)
    ips = set()
    for index, row in data.iterrows():
        # if row['direction'] != '<-' and row['direction'] != '->':
        if row['direction'].strip() in ['<->', '<?>']:
            ips.add(row['src_ip'])
            ips.add(row['dst_ip'])
        # elif row['direction'].strip() == '->':
        else:
            ips.add(row['src_ip'])

    print("UNIQUE IPs: ", len(ips))
    #         temp = row['src_ip']
    #         row['src_ip'] = row['dst_ip']
    #         row['dst_ip'] = temp

    # parse packets and bytes as integers instead of strings
    data['packets'] = data['packets'].astype(int)
    data['duration'] = data['duration'].astype(float)
    # data['bytes'] = data['bytes'].astype(int)
    data['src_bytes'] = data['src_bytes'].astype(int)
    data['dst_bytes'] = data['dst_bytes'].astype(int)

    # add the numerical representation of the categorical data
    data['protocol_num'] = pd.Categorical(data['protocol'], categories=data['protocol'].unique()).codes
    # data['state_num'] = pd.Categorical(data['state'], categories=data['state'].unique()).codes
    data['direction_num'] = pd.Categorical(data['direction'], categories=data['direction'].unique()).codes

    return data


training_sets = [44, 45, 46, 48, 51, 52, 52, 54]
evaluation_sets = [42, 43, 47, 49, 50]
all_sets = list(range(42, 55))
custom_set = [42]
unique_ips = 0
# unique_connections = 0
# for s in range(42, 55):
for s in custom_set:
    dataset = 'CTU-Malware-Capture-Botnet-{}'.format(s)
    # dataset = 'capture20110817.pcap.netflow.labeled'
data = read_and_process_data(dataset)
# unique_connections += len([group for _, group in data.groupby(['src_ip', 'dst_ip'])])
unique_ips += len(data['src_ip'].unique())

print("unique_ips: ", unique_ips)
# print("unique_connections: ", unique_connections)
