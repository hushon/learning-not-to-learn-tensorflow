import argparse
import os
import tensorflow as tf
from model import LNTL

def main():

    # GPU memory growth must be enabled on GeForce RTX GPU systems due to tensorflow bug.
    # Reference: https://github.com/tensorflow/tensorflow/issues/24828

    '''from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    [device.name for device in local_device_protos if device.device_type is 'GPU']'''

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        model = LNTL(sess, args)

        if args.phase == 'train':
            model.train()
        elif args.phase == 'test':
            model.test()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--phase', dest='phase', default='train', help='train, test')
    parser.add_argument('--data_dir', dest='data_dir', default='./dataset/colored_mnist', help='dataset dir')
    parser.add_argument('--log_dir', dest='log_dir', default='./logs', help='log dir')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./checkpoint', help='checkpoint dir')
    parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=100, help='maximum epoch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--image_size', dest='image_size', type=int, default=28, help='pixel size')
    parser.add_argument('--input_c_dim', dest='input_c_dim', type=int, default=3, help='number of channels')
    parser.add_argument('--dim_class', dest='dim_class', type=int, default=10, help='number of class categories')
    parser.add_argument('--dim_bias', dest='dim_bias', type=int, default=8, help='bias dimension')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--loss_lambda', dest='loss_lambda', type=float, default=0.01, help='lambda coeff')
    parser.add_argument('--loss_my', dest='loss_mu', type=float, default=1.0, help='mu coeff')

    args = parser.parse_args()

    '''print args'''
    for k, v in vars(args).items():
        print('{} = {}'.format(k, v))

    main()
