from net import find_caffe
caffe_root = find_caffe.caffe_root
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
# os.chdir('..')

def train_solver(conf):
    s = caffe_pb2.SolverParameter()

    # Set a seed for reproducible experiments:
    # this controls for randomization in training.
    #s.random_seed = 0xCAFFE

    # Specify locations of the train and (maybe) test networks.
    s.train_net = conf.train_net_file
    s.test_net.append(conf.test_net_file)
    s.test_interval = 10000  # Test after every 500 training iterations.
    s.test_iter.append(1)  # Test on 100 batches each time we test.
    s.max_iter = conf.max_iter  # no. of times to update the net (training iterations)
    # s.max_iter = 50000  # no. of times to update the net (training iterations)
    s.type = "AdaGrad"
    s.gamma = 0.1
    s.base_lr = 0.01
    s.weight_decay = 5e-4
    s.lr_policy = 'multistep'
    s.display = 10000
    s.snapshot = 10000
    s.snapshot_prefix = conf.snapshot_prefix
    #s.stepvalue.append(1000000)
    #s.stepvalue.append(300000)
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    s.device_id = 1 # will use the second GPU card
    s.snapshot_format = 0 # 0 is HDF5, 1 is binary
    return s


def train_net(conf, batch_size, class_num, train=True) :
    '''
    :param conf: the data_set_config information, defined in data_info_set.item
    :param batch_size: the batch_size of prototxt
    :param class_num: the class_num of the data_set
    :param train: if True, this net file will be train.prototxt, if False, this net file will be test.prototxt
    :return: file handle
    '''
    weight_param = dict(lr_mult = 1)
    bias_param = dict(lr_mult = 2)
    learned_param = [weight_param, bias_param]
    frozen_param = [dict(lr_mult = 0)] * 2
    finetune_param = [dict(lr_mult = 10), dict(lr_mult = 20)]

    n = caffe.NetSpec()
    if train is True :
        with open(conf.train_data_list_file, 'w') as f :
            f.write(conf.train_data_file)
        n.data, n.label = L.HDF5Data(batch_size = batch_size, source = conf.train_data_list_file, shuffle = True,
                                     ntop = 2)
    else :
        with open(conf.test_data_list_file, 'w') as f :
            f.write(conf.test_data_file)
        n.data, n.label = L.HDF5Data(batch_size = batch_size, source = conf.test_data_list_file, shuffle = True, ntop = 2)
    if conf.use_CK is True:
        n.conv1 = L.Convolution(n.data, kernel_h = conf.CK_kernel_size, kernel_w = 1, num_output = 20,
                                param = learned_param,
                                weight_filler = dict(type = 'gaussian', std = 0.05),
                                bias_filler = dict(type = 'constant', value = 0.1))
    else:
        n.conv1 = L.Convolution(n.data, kernel_h=conf.kernel_size, kernel_w=1, num_output=20,
                                param=learned_param,
                                weight_filler=dict(type='gaussian', std=0.05),
                                bias_filler=dict(type='constant', value=0.1))
    n.bn1 = L.BatchNorm(n.conv1, use_global_stats = 1, in_place = True)
    n.relu1 = L.PReLU(n.bn1, in_place = True)
    n.ip1 = L.InnerProduct(n.relu1, num_output = 100, weight_filler = dict(type = 'gaussian', std = 0.05),
                           param = learned_param,
                           bias_filler = dict(type = 'constant', value = 0.1))
    n.relu2 = L.PReLU(n.ip1, in_place = True)
    n.drop1 = L.Dropout(n.ip1, dropout_ratio = 0.1, in_place = True)
    n.ip2 = L.InnerProduct(n.relu2, num_output = class_num, weight_filler = dict(type = 'gaussian', std = 0.05),
                           param = learned_param,
                           bias_filler = dict(type = 'constant', value = 0.1))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    if train is False:
        n.acc = L.Accuracy(n.ip2, n.label)
    return n.to_proto()


def deploy_net(conf, batch_size, class_num) :
    '''
    :param conf:  the data_set_config information, defined in data_info_set.item
    :param batch_size: the batch_size of prototxt
    :param class_num: the class_num of the data_set
    :param channels: the channels of hyperspectral data, maybe it is 224,448 or 103,206
    :param kernel_size: the kernel_size of the convolution layer, often is 1/9 of the channels
    :return: deploy file handle
    '''
    n = caffe.NetSpec()
    if conf.use_CK is True:
        n.data, n.label = L.DummyData(shape= {'dim' : [batch_size, 1, conf.CK_channels, 1]}, ntop = 2)
        n.conv1 = L.Convolution(n.data, kernel_h=conf.CK_kernel_size, kernel_w=1, num_output=20,
                                weight_filler=dict(type='gaussian', std=0.05),
                                bias_filler=dict(type='constant', value=0.1))
    else:
        n.data, n.label = L.DummyData(shape= {'dim' : [batch_size, 1, conf.channels, 1]}, ntop = 2)
        n.conv1 = L.Convolution(n.data, kernel_h = conf.kernel_size, kernel_w = 1, num_output = 20,
                                weight_filler = dict(type = 'gaussian', std = 0.05),
                                bias_filler = dict(type = 'constant', value = 0.1))
    n.bn1 = L.BatchNorm(n.conv1, use_global_stats = 1, in_place = True)
    n.relu1 = L.PReLU(n.bn1, in_place = True)
    n.ip1 = L.InnerProduct(n.relu1, num_output = 100, weight_filler = dict(type = 'gaussian', std = 0.05),
                           bias_filler = dict(type = 'constant', value = 0.1))
    n.drop1 = L.Dropout(n.ip1, dropout_ratio = 0.1, in_place = True)
    n.relu2 = L.PReLU(n.drop1, in_place = True)
    n.ip2 = L.InnerProduct(n.relu2, num_output = class_num, weight_filler = dict(type = 'gaussian', std = 0.05),
                            bias_filler = dict(type = 'constant', value = 0.1))
    return n.to_proto()


def set_prototxt(conf, test_nums, class_num):
    '''
    :param conf: the data_set_config information, defined in data_info_set.item
    :param test_nums: the number of testset
    :return: nothing
    '''
    with open(conf.train_net_file, 'w') as f :
        f.write(str(train_net(conf = conf, batch_size = 32, class_num = class_num, train = True)))
    with open(conf.test_net_file, 'w') as f :
        f.write(str(train_net(conf = conf, batch_size = 10000, class_num = class_num, train = False)))
    with open(conf.deploy_net_file, 'w') as f :
        f.write(str(deploy_net(conf = conf, batch_size = 16, class_num = class_num)))
    with open(conf.solver_file, 'w') as f :
        f.write(str(train_solver(conf = conf)))
