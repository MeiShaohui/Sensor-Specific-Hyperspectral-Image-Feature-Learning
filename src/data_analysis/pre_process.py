import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import hyperspectral_datasets as HSI


class HSI_preprocess:
    # add zeros to make the channels of the data to 224
    def __init__(self, name, dst_shape):
        self.name = name
        self.dst_shape=dst_shape

    def add_channel(self, data):
        if self.name is 'indian_pines':
            data_add_channel = np.zeros(self.dst_shape)
            print ('After add channel to origin data, the data shape is: ', data_add_channel.shape)
            data_add_channel[:, :, 0:103] = data[:, :, 0:103]
            data_add_channel[:, :, 109:149] = data[:, :, 104:144]
            data_add_channel[:, :, 164:219] = data[:, :, 145:200]
            return data_add_channel
        if self.name is 'salina':
            data_add_channel = np.zeros(self.dst_shape)
            print ('After add channel to origin data, the data shape is: ', data_add_channel.shape)
            data_add_channel[:, :, 0 :107] = data[:, :, 0 :107]
            data_add_channel[:, :, 112 :153] = data[:, :, 107 :148]
            data_add_channel[:, :, 167 :223] = data[:, :, 148 :204]
            return data_add_channel

    def data_add_zero(self, data, patch_size=5):
        """
        Add zeros to make data easy to mean and var
        :param data:
        :param patch_size:
        :return:
        """
        assert data.ndim == 3
        dx = patch_size // 2
        data_add_zeros = np.zeros( (data.shape[0]+2*dx, data.shape[1]+2*dx, data.shape[2]))
        data_add_zeros[dx:-dx, dx:-dx, :] = data
        return data_add_zeros

    def get_mean_data(self, data, patch_size=5, debug=False, var=False):
        """
        Get the mean or var of n*n patch
        :param data:
        :param patch_size:
        :param debug:
        :param var:
        :return:
        """
        assert isinstance(data.flatten()[0], float)
        dx = patch_size // 2
        # add zeros for mirror data
        data_add_zeros = self.data_add_zero(data=data, patch_size=patch_size)
        # get mirror date to calculate boundary pixel
        for i in range(dx):
            data_add_zeros[:, i, :] = data_add_zeros[:, 2 * dx - i, :]
            data_add_zeros[i, :, :] = data_add_zeros[2 * dx - i, :, :]
            data_add_zeros[:, -i - 1, :] = data_add_zeros[:, -(2 * dx - i) - 1, :]
            data_add_zeros[-i - 1, :, :] = data_add_zeros[-(2 * dx - i) - 1, :, :]
        if debug is True:
            print (data_add_zeros)
        data_mean = np.zeros(data.shape)
        data_var =np.zeros(data.shape)
        #get mean and var for evey patch
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                x_start, x_end = x, x+patch_size
                y_start, y_end = y, y+patch_size
                patch = np.array(data_add_zeros[x_start:x_end, y_start:y_end, :])
                data_mean[x, y, :] = np.mean(patch.reshape(patch_size**2, patch.shape[2]), axis = 0)
                if var is True:
                    data_var[x, y, :]=np.std(patch.reshape(patch_size**2, patch.shape[2]), axis = 0)
        if var is False:
            return data_mean
        return np.concatenate((data_mean, data_var), axis = 2)

    def get_patch_data(self, data, patch_size=5):
        """
        :param data: m x n x c
        :param patch_size:
        :return: m x n x patch_size x patch_size x c
        """
        assert isinstance(data.flatten()[0], float)
        dx = patch_size // 2
        # add zeros for mirror data
        data_add_zeros = self.data_add_zero(data=data, patch_size=patch_size)
        # get mirror date to calculate boundary pixel
        for i in range(dx):
            data_add_zeros[:, i, :] = data_add_zeros[:, 2 * dx - i, :]
            data_add_zeros[i, :, :] = data_add_zeros[2 * dx - i, :, :]
            data_add_zeros[:, -i - 1, :] = data_add_zeros[:, -(2 * dx - i) - 1, :]
            data_add_zeros[-i - 1, :, :] = data_add_zeros[-(2 * dx - i) - 1, :, :]
        data_out = np.zeros(list(data.shape[:-1]) + [patch_size, patch_size] + [data.shape[-1],])
        #get mean and var for evey patch
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                x_start, x_end = x, x+patch_size
                y_start, y_end = y, y+patch_size
                patch = np.array(data_add_zeros[x_start:x_end, y_start:y_end, :])
                data_out[x, y, :, :, :] = patch
        return data_out

    def scale_to1(self, data):
        assert data.ndim == 3
        min_value = min(data.flatten())
        if min_value >= 0:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    data[i, j, :] /= np.max(np.abs(data[i, j, :]))
                    data[i, j, :] = data[i, j, :] * 2 - 1
        else:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    data[i, j, :] /= np.max(np.abs(data[i, j, :]))
        return data


#get lable and original data
def set_and_save_indian_pines_proceed_data():
    dataset = HSI.HSIDataSet('indian_pines')
    dataset.get_data()
    dataset.get_labels()
    print ('data shape is: ', dataset.data.shape)  # 145,145,200
    print ('label shape is: ', dataset.labels.shape)  # 145, 145

    data, labels = np.array(dataset.data), np.array(dataset.labels)
    process = HSI_preprocess(name = 'indian_pines', dst_shape = (145,145,224))
    data_add_channel = process.add_channel(data)

    data_3x3_mean = process.get_mean_data(data = data_add_channel, patch_size = 3, var = False)
    data_3x3_mean = process.scale_to1(data_3x3_mean)
    sio.savemat(dataset.dir + '/indian_pines_3x3_mean.mat', {'data' : data_3x3_mean, 'labels' : labels})

    data_5x5_mean = process.get_mean_data(data = data_add_channel, patch_size = 5, var = False)
    data_5x5_mean = process.scale_to1(data_5x5_mean)
    sio.savemat(dataset.dir + '/indian_pines_5x5_mean.mat', {'data' : data_5x5_mean, 'labels' : labels})

    data_3x3_mean_std = process.get_mean_data(data = data_add_channel, patch_size = 3, var = True)
    data_3x3_mean_std = process.scale_to1(data_3x3_mean_std)
    sio.savemat(dataset.dir + '/indian_pines_3x3_mean_std.mat', {'data' : data_3x3_mean_std, 'labels' : labels})

    data_5x5_mean_std = process.get_mean_data(data = data_add_channel, patch_size = 5, var = True)
    data_5x5_mean_std = process.scale_to1(data_5x5_mean_std)
    sio.savemat(dataset.dir + '/indian_pines_5x5_mean_std.mat', {'data' : data_5x5_mean_std, 'labels' : labels})


def set_and_save_salina_proceed_data():
    dataset = HSI.HSIDataSet('salina')
    dataset.get_data()
    dataset.get_labels()
    print ('data shape is: ', dataset.data.shape)  # 145,145,200
    print ('label shape is: ', dataset.labels.shape)  # 145, 145

    data, labels = np.array(dataset.data), np.array(dataset.labels)
    process = HSI_preprocess(name = 'salina', dst_shape=(512, 217, 224))
    data_add_channel = process.add_channel(data)

    data_3x3_mean = process.get_mean_data(data = data_add_channel, patch_size = 3, var = False)
    data_3x3_mean = process.scale_to1(data_3x3_mean)
    sio.savemat(dataset.dir + '/salina_3x3_mean.mat', {'data' : data_3x3_mean, 'labels' : labels})

    data_5x5_mean = process.get_mean_data(data = data_add_channel, patch_size = 5, var = False)
    data_5x5_mean = process.scale_to1(data_5x5_mean)
    sio.savemat(dataset.dir + '/salina_5x5_mean.mat', {'data' : data_5x5_mean, 'labels' : labels})

    data_3x3_mean_std = process.get_mean_data(data = data_add_channel, patch_size = 3, var = True)
    data_3x3_mean_std = process.scale_to1(data_3x3_mean_std)
    sio.savemat(dataset.dir + '/salina_3x3_mean_std.mat', {'data' : data_3x3_mean_std, 'labels' : labels})

    data_5x5_mean_std = process.get_mean_data(data = data_add_channel, patch_size = 5, var = True)
    data_5x5_mean_std = process.scale_to1(data_5x5_mean_std)
    sio.savemat(dataset.dir + '/salina_5x5_mean_std.mat', {'data' : data_5x5_mean_std, 'labels' : labels})


def set_and_save_indian_pines_proceed_1x1_data():
    dataset = HSI.HSIDataSet('indian_pines')
    dataset.get_data()
    dataset.get_labels()
    print ('data shape is: ', dataset.data.shape)  # 145,145,200
    print ('label shape is: ', dataset.labels.shape)  # 145, 145

    data, labels = np.array(dataset.data), np.array(dataset.labels)
    process = HSI_preprocess(name = 'indian_pines', dst_shape = (145,145,224))
    data_add_channel = process.add_channel(data)
    data_1x1 = process.scale_to1(data_add_channel)
    sio.savemat(dataset.dir + '/indian_pines_1x1_mean.mat', {'data' : data_1x1, 'labels' : labels})


def set_and_save_salina_proceed_1x1_data():
    dataset = HSI.HSIDataSet('salina')
    dataset.get_data()
    dataset.get_labels()
    print ('data shape is: ', dataset.data.shape)  # 145,145,200
    print ('label shape is: ', dataset.labels.shape)  # 145, 145

    data, labels = np.array(dataset.data), np.array(dataset.labels)
    process = HSI_preprocess(name = 'salina', dst_shape=(512, 217, 224))
    data_add_channel = process.add_channel(data)
    data_1x1 = process.scale_to1(data_add_channel)
    sio.savemat(dataset.dir + '/salina_1x1_mean.mat', {'data' : data_1x1, 'labels' : labels})


def set_and_save_proceed_HSI_data():
    set_and_save_indian_pines_proceed_data()
    set_and_save_salina_proceed_data()


def set_and_save_proceed_1x1_HSI_data():
    set_and_save_indian_pines_proceed_1x1_data()
    set_and_save_salina_proceed_1x1_data()


if __name__=='__main__':
    set_and_save_proceed_HSI_data()
    set_and_save_proceed_1x1_HSI_data()
    print("hello")
