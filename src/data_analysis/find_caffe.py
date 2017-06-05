# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:54:30 2016

@author: root
"""
try:
    import os, sys
    import caffe
except ImportError:
    caffe_root = os.path.expanduser('~/caffe')
    sys.path.insert(0, caffe_root + '/python')
    import caffe
else:
    caffe_root = os.path.expanduser('~/caffe')


if __name__=='__main__':
    pass
