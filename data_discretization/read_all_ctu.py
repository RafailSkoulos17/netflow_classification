import inspect
import os
import sys

import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from pypackage.flexfringe.utils import preprocess_bidirectional_data, remove_background


def read_and_process_data(dataset):
    dataset_dir = '../data/ctu/'

    # read the data in chunks due to their large size - uncomment the following lines if you want to read them again
    # and store them in a pickle
    preprocess_bidirectional_data(dataset_dir + dataset)
    dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d %H:%M:%S.%f')
    data = pd.concat(
        remove_background(chunk) for chunk in pd.read_csv(dataset_dir + dataset + '_v2',
                                                          chunksize=100000, delimiter=',',
                                                          parse_dates=['date'], date_parser=dateparse))
    data.to_pickle(dataset_dir + 'no_background_' + dataset + '.pkl')
    try:
        os.remove(dataset_dir + dataset + '_v2')
    except OSError:
        pass


scenarios = list(range(42, 55))

for s in scenarios:
    dataset = 'CTU-Malware-Capture-Botnet-{}'.format(s)
    read_and_process_data(dataset)
