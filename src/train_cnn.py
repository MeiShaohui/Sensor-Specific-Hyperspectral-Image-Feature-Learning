import os
import sys
import h5py
import argparse
import net.proto_file as proto_file
import subprocess
import numpy as np
import scipy.io as sio
import data_analysis.find_caffe as find_caffe
import Config.ExpConfigInfo as Config


caffe_root = find_caffe.caffe_root


def train_aviris_10_times(label_unique, args):
    for i in range(5):
        exp_info = Config.ExpConfigInfo(name=args.data_set, label_unique=label_unique,
                                        new_dir_name=args.dst_dir,
                                        gpus=args.gpu, net_name='bn_net', exp_index=i,
                                        spatial_info=args.spatial_info, train_nums=args.train_nums)
        # set hyperparameters
        exp_info.set_data()
        exp_info.max_iter = args.max_iter
        exp_info.set_final_model()
        # train
        proto_file.set_prototxt(exp_info, exp_info.test_nums, exp_info.max_class)
        job_file = 'job_file_gpu_{}.sh'.format(exp_info.gpus)

        with open(job_file, 'w') as f:
            # f.write('cd {}\n'.format(caffe_root))
            f.write(caffe_root + '/build/tools/caffe train \\\n')
            f.write('--solver="{}" \\\n'.format(exp_info.solver_file))
            f.write('--gpu {} 2>&1 | tee {}\n'.format(exp_info.gpus, exp_info.log_file))

        subprocess.check_call('bash {}'.format(job_file), shell=True)

        test_dict = Config.get_y_pred_from_model(model=exp_info, mode='test', score_layer_name='ip2')
        train_dict = Config.get_y_pred_from_model(model=exp_info, mode='train', score_layer_name='ip2')
        test_feature = Config.get_feature_from_model(model=exp_info, mode='test', score_layer_name='ip1')
        train_feature = Config.get_feature_from_model(model=exp_info, mode='train', score_layer_name='ip1')
        sio.savemat(exp_info.result_mat_file, {'train': train_dict, 'test': test_dict, 'train_feature': train_feature,
                                               'test_feature': test_feature})


def train_indian_pines(args):
    label_unique = [2, 3, 5, 6, 8, 10, 11, 12, 14]
    train_aviris_10_times(label_unique, args=args)


def train_salina(args):
    label_unique = range(1, 17)
    train_aviris_10_times(label_unique, args=args)


def train(args):
    if args.data_set == 'indian_pines':
        train_indian_pines(args)
    elif args.data_set == 'salina':
        train_salina(args)
    else:
        raise NameError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train bn net",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--spatial_info', type=str, default='5x5_mean_std',
                        help="1x1_mean', '3x3_mean', '3x3_mean_std', '5x5_mean', '5x5_mean_std")
    parser.add_argument('--gpu', type=int, default=1,
                        help='the number of gpu id, only one number is required')
    parser.add_argument('--dst_dir', type=str, default='bn_net_200',
                        help='the destination dir for the experiments')
    parser.add_argument('--data_set', type=str, default='salina',
                        help='indian_pines, salina')
    parser.add_argument('--max_iter', type=int, default=10000,
                        help='how many iters')
    parser.add_argument('--train_nums', type=float, default=200,
                        help='how many samples for training or how much percents for training, 200 or 0.1')
    args = parser.parse_args()
    train(args=args)