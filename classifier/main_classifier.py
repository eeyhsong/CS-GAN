"""
use for CNN_basis.py
test the acc of 9 subject
cross validation data
"""


# from preprocess import *
# from CNN_basic import cnn

# from pre_fake_csp import  *
from pre_cla import *
# from CNN_basic_csp_cv_9 import cnn
from CNN import cnn
import matplotlib.pyplot as plt
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,4,7'
# devices = ['gpu:0,1,2,3,4']

# sub_index = 2
for sub_index in range(9):
    sub_index += 1
    # sub_index = 9
    # data_train, label_train, data_test, label_test = aug_data(sub_index)
    data_train, label_train, data_test, label_test = split_subject(sub_index)
    # data_train, label_train, data_test, label_test = split_half(sub_index)
    # data_train, label_train, data_test, label_test = split_cross(sub_index)

    '''
    # used for new structure designed by pytorch
    data_train = np.squeeze(data_train).swapaxes(1, 2)
    data_train = np.expand_dims(data_train, axis=1)
    data_test = np.squeeze(data_test).swapaxes(1, 2)
    data_test = np.expand_dims(data_test, axis=1)
    label_train = np.squeeze(label_train)
    label_test = np.squeeze(label_test)
    '''

    # cnn(sub_index, data_train, label_train, data_test, label_test,
    #     conv_layers=2, conv_sizes=(64, 64), fc_layers=3, fc_sizes=(1024, 512, 256))
    # cnn(sub_index, data_train, label_train, data_test, label_test,
    #     conv_layers=2, conv_sizes=(64, 128, 256, 512), fc_layers=3, fc_sizes=(1024, 512, 256))
    cnn(sub_index, data_train, label_train, data_test, label_test)




