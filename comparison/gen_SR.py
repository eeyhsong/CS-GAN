import numpy as np
from pre_for_adding_noise import *

for sub_index in range(9):
    sub_index += 1
    data_train, label_train, data_test, label_test = split_subject(sub_index)

    for nclass in range(0, 4):

        class_sample_index = np.argwhere(label_train == nclass)
        data_train_class = data_train[class_sample_index, :, :]  # data for one subject one class
        label_train_class = label_train[class_sample_index]  # label for one subject one class

        data_train_class = np.squeeze(data_train_class)
        data_train_class = data_train_class.swapaxes(1, 2)
        label_train_class = np.squeeze(label_train_class)

        data_train_class = np.expand_dims(data_train_class, axis=1)
        gen_data = np.zeros((1250, 1, 22, 1000))
        for gen_index in range(1250):
            select_index = np.random.randint(0, len(data_train_class), 8)
            for seg_index in range(8):
                gen_data[gen_index, 0, :, 125*seg_index:125*(seg_index+1)] = \
                    data_train_class[select_index[seg_index], 0, :, 125*seg_index:125*(seg_index+1)]

        # save generated data
        np.save("/home/syh/Documents/MI/experiments/gen_data/gen_justT_SR/S" + str(sub_index) + "_class" + str(nclass+1) + ".npy", gen_data)


