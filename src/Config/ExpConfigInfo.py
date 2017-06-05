import os
import sys
import h5py
import numpy as np
import scipy.io as sio
import find_caffe as find_caffe
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '..'))
import data_analysis.get_feature_from_model as feature
import data_analysis.pre_process as pre
import data_analysis.hyperspectral_datasets as HSI
import data_analysis.train_test_split as train_test_split

caffe_root = find_caffe.caffe_root


class ExpConfigInfo(object):
    def __init__(self, name, label_unique, new_dir_name, gpus=0, net_name='bn_net', exp_index=0, spatial_info='1x1_mean', train_nums=200):
        self.name = name

        self.proto_dir = '../result/{}/{}/proto'.format(self.name, new_dir_name)
        self.model_dir = '../result/{}/{}/model'.format(self.name, new_dir_name)
        self.result_dir = '../result/{}/{}/result'.format(self.name, new_dir_name)

        self.train_net_file = '{}/{}_{}_train.prototxt'.format(self.proto_dir, self.name, spatial_info)
        self.test_net_file = '{}/{}_{}_test.prototxt'.format(self.proto_dir, self.name, spatial_info)
        self.deploy_net_file = '{}/{}_{}_deploy.prototxt'.format(self.proto_dir, self.name, spatial_info)
        self.solver_file = '{}/{}_{}_solver.prototxt'.format(self.proto_dir, self.name, spatial_info)
        self.train_data_list_file = '{}/{}_train_file.txt'.format(self.proto_dir, spatial_info)
        self.test_data_list_file = '{}/{}_test_file.txt'.format(self.proto_dir, spatial_info)

        self.data_dir = os.path.expanduser('../hyperspectral_datas/{}/data/'.format(self.name))
        self.train_data_file = self.data_dir + '{}_train_{}.h5'.format(self.name, spatial_info)
        self.test_data_file = self.data_dir + '{}_test_{}.h5'.format(self.name, spatial_info)

        self.snapshot_prefix = '{}/{}_models_time_{}'.format(self.model_dir, spatial_info, exp_index)
        self.deploy_file = os.getcwd() + '/' + self.deploy_net_file

        # set hyperparameters
        self.channels = 224
        self.CK_channels = 448
        self.kernel_size = 24
        self.CK_kernel_size = 48
        self.max_iter = 30000
        self.train_nums = train_nums
        self.use_CK = spatial_info in ['3x3_mean_std', '5x5_mean_std']

        # save results and final model
        self.gpus = gpus
        self.net_name = net_name
        self.label_unique = label_unique
        self.spatial_info = spatial_info
        self.log_file = "{}/{}_{}_{}_{}.log".format(os.getcwd() + '/' + self.result_dir, self.net_name, self.name,
                                                    self.spatial_info,
                                                    exp_index)
        self.result_mat_file = self.result_dir + '/{}_pred_'.format(exp_index) + spatial_info + '_model_{}.mat'.format(self.name)
        self.result_dat_file = self.result_dir + '/{}_pred_'.format(exp_index) + spatial_info + '_model_{}.dat'.format(self.name)
        self.final_model = ''
        self.test_nums = 0
        self.max_class = 0

    def set_data(self):
        mkdir_if_not_exist(self.proto_dir)
        mkdir_if_not_exist(self.model_dir)
        mkdir_if_not_exist(self.result_dir)
        self.test_nums, self.max_class = get_train_test_data(label_unique=self.label_unique,
                                                             dataset_name=self.name, spatial_info=self.spatial_info,
                                                             train_nums=self.train_nums, data_set_dir=self.data_dir)

    def set_final_model(self):
        self.final_model = os.getcwd() + '/' + self.snapshot_prefix + '_iter_{}.caffemodel.h5'.format(
            self.max_iter)


def mkdir_if_not_exist(the_dir):
    if not os.path.isdir(the_dir) :
        os.makedirs(the_dir)


def get_train_test_data(label_unique, dataset_name = 'indian_pines', spatial_info='5x5_mean_std', train_nums=200, data_set_dir=''):
    assert dataset_name in ['indian_pines', 'salina']
    assert spatial_info in ['1x1_mean', '3x3_mean', '3x3_mean_std', '5x5_mean', '5x5_mean_std']

    class data_set_info:pass
    data_set_info.data = sio.loadmat(data_set_dir + '/' + dataset_name + '_' + spatial_info + '.mat')['data']
    data_set_info.labels = sio.loadmat(data_set_dir + '/' + dataset_name + '_' + spatial_info + '.mat')['labels']
    data_set_info.h5train = data_set_dir + '/' + dataset_name + '_train_' + spatial_info + '.h5'
    data_set_info.h5test = data_set_dir + '/' + dataset_name + '_test_' + spatial_info + '.h5'
    (train_label, train_index, train_data), (test_label, test_index, test_data) = train_test_split.train_test_split(
        data_set_info.data, data_set_info.labels,
        label_unique=label_unique,
        train=train_nums)

    put_data_to_h5file({'data': train_data, 'labels': train_label, 'index': train_index}, data_set_info.h5train)
    put_data_to_h5file({'data': test_data, 'labels': test_label, 'index': test_index}, data_set_info.h5test)
    return len(test_label), max(label_unique)+1


def put_data_to_h5file(data, file_name, isRPCA=False):
    if isRPCA:
        write_data = data['data']
    else:
        write_data = data['data'].reshape(data['data'].shape[0], 1, data['data'].shape[1], 1)
    write_label = data['labels'].reshape(data['labels'].shape[0], 1)
    write_index = data['index']
    if os.path.exists(file_name):
        os.remove(file_name)
    f = h5py.File(file_name, 'w')
    f.create_dataset('data',shape = write_data.shape, dtype=np.float32, data = write_data)
    f.create_dataset('label', shape=write_label.shape, dtype=np.float32, data=write_label)
    f.create_dataset('index', dtype=np.float32, data=write_index)
    f.close()


def get_y_pred_from_model(model, mode='test', score_layer_name = 'ip2'):
    assert os.path.exists(model.deploy_file) and os.path.exists(model.final_model)
    model_feature = feature.GetFeatureFromCaffe(deploy_file=model.deploy_file, pretrained_model=model.final_model, score_layer_name=score_layer_name)
    if mode is 'test':
        assert os.path.exists(model.test_data_file)
        model_feature.get_h5_data(model.test_data_file)
    elif mode is 'train':
        assert os.path.exists(model.train_data_file)
        model_feature.get_h5_data(model.train_data_file)
    model_feature.get_metric()
    return {
        'classify_report' : model_feature.classify_report,
        'confusion_matrix' : model_feature.confusion_matrix,
        'y_true' : model_feature.y_true,
        'y_pred' : model_feature.y_pred,
        'y_index' : model_feature.index,
        'OA' : model_feature.overall_accuracy,
        'AA' : model_feature.average_accuracy,
        'ACC' : model_feature.acc_for_each_class
    }


def get_feature_from_model(model, mode='test', score_layer_name = 'ip1'):
    assert os.path.exists(model.deploy_file) and os.path.exists(model.final_model)
    model_feature = feature.GetFeatureFromCaffe(deploy_file=model.deploy_file, pretrained_model=model.final_model, score_layer_name=score_layer_name)
    if mode is 'test':
        assert os.path.exists(model.test_data_file)
        model_feature.get_h5_data(model.test_data_file)
    elif mode is 'train':
        assert os.path.exists(model.train_data_file)
        model_feature.get_h5_data(model.train_data_file)
    model_feature.get_y_pred()
    return {
        'y_feature' : model_feature.feature,
        'y_index' : model_feature.index,
    }