# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 20:54:43 2016

@author: root
"""

import os
import sys
import h5py
import find_caffe
caffe = find_caffe.caffe
import numpy as np
import scipy.io as sio
from sklearn import metrics


class GetFeatureFromCaffe(caffe.Net):
    def __init__(self, deploy_file, pretrained_model, score_layer_name='ip2'):
        self.deploy_file = deploy_file
        self.pretrained_model = pretrained_model
        caffe.set_device(0)  # if we have multiple GPUs, pick the first one
        caffe.set_mode_gpu()
        self.net = caffe.Net(self.deploy_file, self.pretrained_model, caffe.TEST)
        self.score_layer_name=score_layer_name

    def get_h5_data(self, data_in_name):
        f = h5py.File(data_in_name, 'r')
        self.data = f['data'][:]
        self.label = f['label'][:]
        self.index = f['index'][:]
        f.close()

    def set_data(self, data, label):
        if data.ndim == 3 and label.ndim==2:
            data = data.reshape([data.shape[0]*data.shape[1], 1, data.shape[2], 1])
            label = label.flatten()
        self.data = data
        self.label = label

    def get_ip1(self):
        ip1_dim = self.net.blobs['ip1'].data.shape[-1]
        self.ip1_data = np.zeros((self.label.shape[0], ip1_dim), dtype=np.float32)
        batch_size = self.net.blobs['data'].data.shape[0]
        # put it into the CaffeNet
        for i in np.arange(self.label.shape[0] / batch_size):
            self.net.blobs['data'].data[:] = self.data[i * batch_size:i * batch_size + batch_size, :, :, :]
            self.net.forward()
            self.ip1_data[i * batch_size:i * batch_size + batch_size, :] = self.net.blobs['ip1'].data.reshape(
                batch_size, ip1_dim)

        if self.label.shape[0] % batch_size:
            length = self.label.shape[0] % batch_size
            self.net.blobs['data'].data[:] = self.data[self.label.shape[0] - batch_size:, :, :, :]
            self.net.forward()
            self.ip1_data[self.label.shape[0] - batch_size:, :] = self.net.blobs['ip1'].data.reshape(batch_size,
                                                                                                     ip1_dim)

    def get_y_pred(self):
        layer_name = self.net.outputs[0]
        class_num = self.net.blobs[self.score_layer_name].data.shape[-1]
        self.feature = np.zeros((self.label.shape[0], class_num), dtype = np.float32)
        batch_size = self.net.blobs['data'].data.shape[0]
        # put it into the CaffeNet
        for i in np.arange(self.label.shape[0] / batch_size) :
            self.net.blobs['data'].data[:] = self.data[i * batch_size :i * batch_size + batch_size, :, :, :]
            print i
            self.net.forward()
            self.feature[i * batch_size:i * batch_size + batch_size, :] = self.net.blobs[
                self.score_layer_name].data.reshape(batch_size,
                                                    class_num)
        if self.label.shape[0] % batch_size :
            self.net.blobs['data'].data[:] = self.data[self.label.shape[0] - batch_size :, :, :, :]
            self.net.forward()
            self.feature[self.label.shape[0] - batch_size:, :] = self.net.blobs[self.score_layer_name].data.reshape(
                batch_size, class_num)

    def get_metric(self):
        self.get_y_pred()
        #self.get_ip1()
        self.y_true = self.label
        self.y_pred = self.feature.argmax(1)
        self.classify_report = metrics.classification_report(self.y_true, self.y_pred)
        self.confusion_matrix = metrics.confusion_matrix(self.y_true, self.y_pred)
        self.overall_accuracy = metrics.accuracy_score(self.y_true, self.y_pred)
        self.acc_for_each_class = metrics.precision_score(self.y_true, self.y_pred, average=None)
        self.average_accuracy = np.mean(self.acc_for_each_class)
        print metrics.accuracy_score(self.y_true, self.y_pred)


if __name__ == '__main__' :
    class ip: pass


    pretrained_model = '/home/jhm/2016_TGRS/train_job/src/../result/salina/finetune_from_indian_model_ip_cls/model/5x5_mean_std_models_time_0_iter_500000.caffemodel.h5'
    deploy_file = '/home/jhm/2016_TGRS/train_job/src/../result/salina/finetune_from_indian_model_ip_cls/proto/salina_5x5_mean_std_test.prototxt'
    # pretrained_model = '/home/jhm/2016_TGRS/train_job/src/../result/salina/bn_net/model/5x5_mean_std_models_time_0_iter_1000000.caffemodel.h5'
    test_net = caffe.Net(deploy_file,pretrained_model, caffe.TEST)
    accuracy = 0
    for it in xrange(1):
        accuracy += test_net.forward()['acc']
        print it
    accuracy /= 48726.0
    print 123