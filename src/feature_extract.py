import os
import sys
import stat
import h5py
import time
import shutil
import subprocess
import numpy as np
import scipy.io as sio
from data_analysis import find_caffe
# import caffe
import data_analysis.get_feature_from_model as feature

caffe_root = find_caffe.caffe_root


def mkdir_if_not_exist(the_dir):
    if not os.path.isdir(the_dir) :
        os.makedirs(the_dir)


def get_indian_pines_features_from_indian_pines_model():
    for i in range(10):
        class data: pass

        data.data_dir = os.path.expanduser('../hyperspectral_datas/indian_pines/data/')
        data.data_5x5_mean_std = sio.loadmat(data.data_dir + '/indian_pines_5x5_mean_std.mat')['data']
        data.labels_5x5_mean_std = sio.loadmat(data.data_dir + '/indian_pines_5x5_mean_std.mat')['labels']
        data.result_dir = '../result/indian_pines/bn_net_200/feature'
        mkdir_if_not_exist(data.result_dir)
        data.result_file = data.result_dir + '/ip_feature_ip_model_{}.mat'.format(i)
        data.iters = 2000000

        pretrained_model = data.result_dir + '/../model/5x5_mean_std_models_time_{}_iter_{}.caffemodel.h5'.format(i,
                                                                                                                  data.iters)
        deploy_file = data.result_dir + '/../proto/indian_pines_5x5_mean_std_deploy.prototxt'

        getFeature = feature.GetFeatureFromCaffe(deploy_file=deploy_file, pretrained_model=pretrained_model)
        getFeature.set_data(data.data_5x5_mean_std, data.labels_5x5_mean_std)
        getFeature.get_ip1()

        data.result_dict = {'data': getFeature.ip1_data, 'labels': getFeature.label}
        sio.savemat(data.result_file, data.result_dict)


def get_salina_features_from_salina_model():
    for i in range(10):
        class data: pass

        data.data_dir = os.path.expanduser('~/hyperspectral_datas/salina/data/')
        data.data_5x5_mean_std = sio.loadmat(data.data_dir + '/salina_5x5_mean_std.mat')['data']
        data.labels_5x5_mean_std = sio.loadmat(data.data_dir + '/salina_5x5_mean_std.mat')['labels']
        data.result_dir = '../result/salina/bn_net_200/feature'
        mkdir_if_not_exist(data.result_dir)
        data.result_file = data.result_dir + '/salina_feature_salina_5x5_mean_std_model_{}.mat'.format(i)
        data.iters = 2000000

        pretrained_model = data.result_dir + '/../model/5x5_mean_std_models_time_{}_iter_{}.caffemodel.h5'.format(i,
                                                                                                                  data.iters)
        deploy_file = data.result_dir + '/../proto/salina_5x5_mean_std_deploy.prototxt'

        getFeature = feature.GetFeatureFromCaffe(deploy_file=deploy_file, pretrained_model=pretrained_model)
        getFeature.set_data(data.data_5x5_mean_std, data.labels_5x5_mean_std)
        getFeature.get_ip1()

        data.result_dict = {'data': getFeature.ip1_data, 'labels': getFeature.label}
        sio.savemat(data.result_file, data.result_dict)


def get_indian_pines_features_from_salina_model():
    for i in range(10):
        class data: pass

        data.data_dir = os.path.expanduser('../hyperspectral_datas/indian_pines/data/')
        data.data_5x5_mean_std = sio.loadmat(data.data_dir + '/indian_pines_5x5_mean_std.mat')['data']
        data.labels_5x5_mean_std = sio.loadmat(data.data_dir + '/indian_pines_5x5_mean_std.mat')['labels']
        data.result_dir = '../result/salina/bn_net_200/feature'
        mkdir_if_not_exist(data.result_dir)
        data.result_file = data.result_dir + '/ip_feature_salina_model_{}.mat'.format(i)
        data.iters = 2000000

        pretrained_model = data.result_dir + '/../model/5x5_mean_std_models_time_{}_iter_{}.caffemodel.h5'.format(i,
                                                                                                                  data.iters)
        deploy_file = data.result_dir + '/../proto/salina_5x5_mean_std_deploy.prototxt'

        getFeature = feature.GetFeatureFromCaffe(deploy_file=deploy_file, pretrained_model=pretrained_model)
        getFeature.set_data(data.data_5x5_mean_std, data.labels_5x5_mean_std)
        getFeature.get_ip1()

        data.result_dict = {'data': getFeature.ip1_data, 'labels': getFeature.label}
        sio.savemat(data.result_file, data.result_dict)


def get_salina_features_from_indian_pines_model():
    for i in range(10):
        class data: pass

        data.data_dir = os.path.expanduser('../hyperspectral_datas/salina/data/')
        data.data_5x5_mean_std = sio.loadmat(data.data_dir + '/salina_5x5_mean_std.mat')['data']
        data.labels_5x5_mean_std = sio.loadmat(data.data_dir + '/salina_5x5_mean_std.mat')['labels']
        data.result_dir = '../result/indian_pines/bn_net_200/feature'
        mkdir_if_not_exist(data.result_dir)
        data.result_file = data.result_dir + '/salina_feature_ip_model_{}.mat'.format(i)
        data.iters = 2000000

        pretrained_model = data.result_dir + '/../model/5x5_mean_std_models_time_{}_iter_{}.caffemodel.h5'.format(i,
                                                                                                                  data.iters)
        deploy_file = data.result_dir + '/../proto/indian_pines_5x5_mean_std_deploy.prototxt'

        getFeature = feature.GetFeatureFromCaffe(deploy_file=deploy_file, pretrained_model=pretrained_model)
        getFeature.set_data(data.data_5x5_mean_std, data.labels_5x5_mean_std)
        getFeature.get_ip1()

        data.result_dict = {'data': getFeature.ip1_data, 'labels': getFeature.label}
        sio.savemat(data.result_file, data.result_dict)


if __name__ == '__main__':
    start = time.time()
    get_indian_pines_features_from_indian_pines_model()
    get_salina_features_from_salina_model()
    get_indian_pines_features_from_salina_model()
    get_salina_features_from_indian_pines_model()
    end = time.time()
    print(end - start)
