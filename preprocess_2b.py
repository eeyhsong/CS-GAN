"""
preprocess for eeg data to generate new data with the constrains of CSP conditions
for cross validation

use for BCI competition IV 2b
"""


import scipy.io
import numpy as np
import matplotlib.pyplot as plt


# get the data from .m files
# standardize

def import_data(sub_index, datatype):
    data, label = [], []
    # path = '/home/syh/Documents/MI/data/cv_data2/A0'
    path = '/home/syh/Documents/MI/data/dataset2b_process/T_separate/B0'
    tmp = scipy.io.loadmat(path + str(sub_index) + datatype + '.mat')
    data_one_subject = tmp['data']
    csp_data_one_subject = tmp['csp_data']
    data = np.transpose(data_one_subject, (2, 1, 0))
    csp_data = np.transpose(csp_data_one_subject, (2, 1, 0))
    # data.append(data_one_subject)

    label_one_subject = tmp['label']
    label = label_one_subject.T[0]
    # label.append(label_one_subject)

    return data, csp_data, label  # (288, 22, 1000) (288,)


# one subject: split the data of one subject to train set and val set (actually, T and E)
def split_subject(sub_index, standardize=True):
    # path = '/home/syh/Documents/MI/data/cv_data2/A0'
    path = '/home/syh/Documents/MI/data/dataset2b_process/T_separate/B0'
    tmp = scipy.io.loadmat(path + str(sub_index) + 'T.mat')
    Cov = tmp['Cov']
    Dis_mean = tmp['Dis_mean']
    Dis_std = tmp['Dis_std']
    P = tmp['PP']
    B = tmp['BB']
    Wb = tmp['Wb']

    t_data, t_csp_data, t_label = import_data(sub_index, datatype='T')
    e_data, e_csp_data, e_label = import_data(sub_index, datatype='E')
    # shuffle the train data
    num = np.random.permutation(len(t_data))

    data_train = t_data[num, :, :]
    csp_data_train = t_csp_data[num, :, :]
    label_train = t_label[num]

    data_test = e_data
    csp_data_test = e_csp_data
    label_test = e_label
    # standardize
    # mean and var as the first dim
    # in actual use, we just know the distribution of the training data

    data_train = np.transpose(data_train, (0, 2, 1))
    csp_data_train = np.transpose(csp_data_train, (0, 2, 1))
    data_test = np.transpose(data_test, (0, 2, 1))
    csp_data_test = np.transpose(csp_data_test, (0, 2, 1))

    # change to 0-3 for train
    label_train -= 1
    label_test -= 1

    return data_train, csp_data_train, label_train, data_test, csp_data_test, label_test, \
           Cov, Dis_mean, Dis_std, P, B, Wb


# half cross subject: all T as the training data, one E as the test data
def split_half(sub_index, standardize=True):
    data_train = []
    label_train = []
    for i in range(9):
        t_data, t_label = import_data(i+1, datatype='T')
        data_train.append(t_data)
        label_train.append(t_label)

    data_train = np.concatenate((data_train[0], data_train[1], data_train[2], data_train[3], data_train[4],
                                data_train[5], data_train[6], data_train[7], data_train[8]))
    label_train = np.concatenate((label_train[0], label_train[1], label_train[2], label_train[3], label_train[4],
                                 label_train[5], label_train[6], label_train[7], label_train[8]))

    num = np.random.permutation(len(data_train))
    data_train = data_train[num, :, :]
    label_train = label_train[num]

    data_test, label_test = import_data(sub_index, datatype='E')

    # standardize
    '''
    mean = data_train.mean(0)
    var = np.sqrt(data_train.var(0))
    if standardize:
        data_train -= mean  # distribute at 0
        data_train /= var  # make the var to 1
        data_test -= mean
        data_test /= var
    '''

    data_train = np.transpose(data_train, (0, 2, 1))
    data_test = np.transpose(data_test, (0, 2, 1))

    label_train -= 1
    label_test -= 1

    return data_train, label_train, data_test, label_test


# cross subject: 8 T data of 8 subjects as the training data, one E data of another subject as the test
def split_cross(sub_index, standardize=True):
    data_train = []
    label_train = []
    for i in range(9):
        if i != sub_index-1:
            t_data, t_label = import_data(i+1, datatype='T')
            data_train.append(t_data)
            label_train.append(t_label)

    data_train = np.concatenate((data_train[0], data_train[1], data_train[2], data_train[3],
                                 data_train[4], data_train[5], data_train[6], data_train[7]))
    label_train = np.concatenate((label_train[0], label_train[1], label_train[2], label_train[3],
                                  label_train[4], label_train[5], label_train[6], label_train[7]))

    num = np.random.permutation(len(data_train))
    data_train = data_train[num, :, :]
    label_train = label_train[num]

    data_test, label_test = import_data(sub_index, datatype='E')

    # standardize
    mean = data_train.mean(0)
    var = np.sqrt(data_train.var(0))
    if standardize:
        data_train -= mean  # distribute at 0
        data_train /= var  # make the var to 1
        data_test -= mean
        data_test /= var

    data_train = np.transpose(data_train, (0, 2, 1))
    data_test = np.transpose(data_test, (0, 2, 1))

    label_train -= 1
    label_test -= 1

    return data_train, label_train, data_test, label_test


'''
def main():
    split_cross()


if __name__ == "__main__":
    main()
'''
