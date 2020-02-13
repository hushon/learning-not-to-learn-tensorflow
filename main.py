import argparse
import os
import tensorflow as tf
from model import LNTL

def main():

    # GPU memory growth must be enabled on GeForce RTX GPU systems due to tensorflow bug.
    # Reference: https://github.com/tensorflow/tensorflow/issues/24828
'''
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    [device.name for device in local_device_protos if device.device_type is 'GPU']
'''
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
    parser.add_argument('--data_dir', dest='data_dir', default=None, help='directory of the dataset')
    parser.add_argument('--log_dir', dest='log_dir', default='./logs', help='tf.summary log directory')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./checkpoint', help='checkpoint directory')
    parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=200, help='# of epoch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
    parser.add_argument('--image_size', dest='image_size', type=int, default=256, help='resize height/width to given image size')
    parser.add_argument('--input_c_dim', dest='input_c_dim', type=int, default=1, help='# of input image channels')
    parser.add_argument('--output_c_dim', dest='output_c_dim', type=int, default=1, help='# of output image channels')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
    args = parser.parse_args()

    '''Print argument values'''
    print('====== Arguments ======')
    for key in vars(args):
        print('{} = {}'.format(key, vars(args)[key]))
    print('=======================')

    main()
