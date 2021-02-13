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
        re_time = int(1250/len(data_train_class))
        gen_data = data_train_class
        for rt in range(re_time):
            noise = np.random.normal(0, 0.001, data_train_class.shape)
            gen_data = np.concatenate((gen_data, data_train_class + noise), axis=0)

        # save generated data
        np.save("/home/syh/Documents/MI/experiments/gen_data/gen_justT_noise2/S" + str(sub_index) + "_class" + str(nclass+1) + ".npy", gen_data)


