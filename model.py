import os
from tkinter import Tk, filedialog
import time
from glob import glob
from datetime import datetime

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import utils
import network
import data_getter
import ckptsaver
import ops
from dataset import Dataset

class LNTL:
    def __init__(self, sess, args):

        self.sess = sess
        self.args = args
        self.dtype = dtype

        self._build_model()

    def _build_model(self):
        '''Defines TensorFlow graph.'''

        args = self.args

        print(' [*] Building tensorflow graph')

        '''main graph'''
        datashape = [None, args.image_size, args.image_size, args.input_c_dim]
        dim_class = 10
        dim_bias = 3
        input_image = tf.placeholder(tf.uint8, shape=datashape, name='input_image')
        label_class = tf.placeholder(tf.uint8, shape=[None, dim_class], name='label_class')
        label_bias = tf.placeholder(tf.uint8, shape=[None, dim_bias], name='label_bias')

        feature = network.f(input_image, name='feature_extractor')
        output_class, output_class_logits = network.g(feature, name='class_predictor')
        output_bias, output_bias_logits = network.h(feature, name='bias_predictor')

        '''loss'''
        lambda_loss = 0.1
        with tf.variable_scope("loss"):
            loss_classifier = tf.nn.softmax_cross_entropy_with_logits(
                labels=label_class,
                logits=output_class_logits)
            loss_bias = tf.nn.softmax_cross_entropy_with_logits(
                labels=label_bias,
                logits=output_bias_logits)
            loss = loss_classifier - lambda_loss*loss_bias

        '''optimizer'''
        f_vars = [var for var in tf.trainable_variables() if 'feature_extractor' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'class_predictor' in var.name]
        h_vars = [var for var in tf.trainable_variables() if 'bias_predictor' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op_classifier = tf.train.AdamOptimizer(args.lr).minimize(loss, var_list=f_vars + g_vars)
            train_op_bias = tf.train.AdamOptimizer(args.lr).maximize(loss, var_list=h_vars)

        global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=tf.constant(0), trainable=False)
        global_epoch = tf.get_variable('global_epoch', dtype=tf.int32, initializer=tf.constant(0), trainable=False)
        increment_global_step = tf.assign_add(global_step, 1)
        increment_global_epoch = tf.assign_add(global_epoch, 1)

        '''Define summary'''
        input_image_summary = tf.summary.image("input_image", input_image)
        loss_summary = tf.summary.scalar("loss", tf.reduce_mean(loss))
        with tf.variable_scope("f_weights"):
            f_weights_summary = tf.summary.merge([tf.summary.histogram(var.name, var) for var in f_vars])
        with tf.variable_scope("g_weights"):
            g_weights_summary = tf.summary.merge([tf.summary.histogram(var.name, var) for var in g_vars])
        with tf.variable_scope("h_weights"):
            h_weights_summary = tf.summary.merge([tf.summary.histogram(var.name, var) for var in h_vars])
        summary_op = tf.summary.merge_all()

        self.input_image = input_image
        self.label_class = label_class
        self.label_bias = label_bias

        self.train_op_classifier = train_op_classifier
        self.train_op_bias = train_op_bias

        self.summary_op = summary_op

        self.global_step = global_step
        self.global_epoch = global_epoch
        self.increment_global_step = increment_global_step
        self.increment_global_epoch = increment_global_epoch

    def train(self):
        """Train pix2pix"""
        sess = self.sess
        args = self.args
        summary_writer_train = self.summary_writer_train
        summary_writer_val = self.summary_writer_val

        data = self.data
        train_init_op = self.train_init_op

        is_training = self.is_training
        d_loss = self.d_loss
        g_dice_score = self.g_dice_score
        g_loss = self.g_loss
        g_vars = self.g_vars
        d_train_op = self.d_train_op
        g_train_op = self.g_train_op
        summary_op = self.summary_op
        global_step = self.global_step
        global_epoch = self.global_epoch
        increment_global_step = self.increment_global_step
        increment_global_epoch = self.increment_global_epoch

        # saver
        saver = tf.train.Saver(max_to_keep=1)

        # summary writer
        summary_writer = tf.summary.FileWriter(os.path.normpath(args.log_dir), sess.graph)

        ## restore checkpoint or initialize.
        load_dir = os.path.normpath(args.ckpt_dir)
        try: 
            ckptsaver.load_checkpoint(sess, saver, load_dir)
        except tf.errors.NotFoundError: 
            sess.run(tf.global_variables_initializer())
            print(" [*] initialized variables")

        global_step = sess.run(global_step)
        global_epoch = sess.run(global_epoch)

        ## load datasets
        root = Tk()
        filepath = filedialog.askopenfilename(title='Select file', 
                                              filetypes = (("numpy files","*.npy"),("all files","*.*")))
        filepath = os.path.normpath(filepath)
        root.withdraw()

        dataset = np.load(filepath, allow_pickle=True, encoding='latin1')
        dataset = dataset.item()

        ## preprocess dataset
        train_bias = []
        for x in dataset['train_image']:
            r = x[..., 0].max()
            g = x[..., 1].max()
            b = x[..., 2].max()
            train_bias.append([r,g,b])
        train_bias = np.array(train_bias)
        dataset.update({'train_bias': train_bias})

        ## training loop
        for epoch in tqdm(range(global_epoch, args.max_epoch)):

            for i in range(len(dataset['train_image'])):

                # train step
                sess.run(self.train_op_classifier,
                         feed_dict={self.input_image: dataset['train_image'][i:i+1],
                                    self.label_class: dataset['train_label'][i:i+1]})
                sess.run(self.train_op_bias,
                         feed_dict={self.input_image: dataset['train_image'][i:i+1],
                                    self.label_bias: dataset['train_bias'][i:i+1]})
                summary_str = sess.run(self.summary_op,
                                       feed_dict={self.input_image: dataset['train_image'][i:i+1],
                                                  self.label_class: dataset['train_label'][i:i+1],
                                                  self.label_bias: dataset['train_bias'][i:i+1]})

                # write summary
                summary_writer.add_summary(summary_str, global_step)

                # increment global step
                global_step = sess.run(increment_global_step)

            # increment global epoch
            global_epoch = sess.run(increment_global_epoch)

    def test(self):
        args = self.args
        sess = self.sess
        summary_writer_test = self.summary_writer_test

        data = self.data
        test_init_op = self.test_init_op

        fake_B = self.fake_B
        is_training = self.is_training

        ## for profiling
        if args.enable_profile:
            from tensorflow.python.client import timeline
            timeline_dir = os.path.normpath(args.timeline_dir)
            if not os.path.exists(timeline_dir): os.makedirs(timeline_dir)

        """Test pix2pix"""
        ## restore checkpoint or initialize.
        saver = tf.train.Saver(max_to_keep=1)
        load_dir = os.path.normpath(args.ckpt_dir)
        try: 
            ckptsaver.load_checkpoint(sess, saver, load_dir)
        except tf.errors.NotFoundError: 
            sess.run(tf.global_variables_initializer())
            print(" [*] initialized variables")

        ## load test dataset
        # test_data_dir = os.path.join(args.data_dir, 'test/*.png')
        # testset = data_getter.load_dataset(test_data_dir, resize=(args.image_size,args.image_size))
        # test_data_dir = os.path.join(args.data_dir, 'test')

        root = Tk()
        test_data_dir = filedialog.askdirectory(title='Select folder containing feature images')
        root.withdraw()

        # testset = data_getter.load_dataset_v2(test_data_dir, resize=(args.image_size,args.image_size))
        testset = Dataset().read_testset(test_data_dir, resize=(args.image_size, args.image_size)).dataset
        testset = np.concatenate([testset['feature'], testset['label']], -1)
        sess.run(test_init_op, feed_dict={data: testset})

        test_rate = 0.0
        idx = 0

        while True:
            try:
                starttime = time.time()

                if args.enable_profile: 
                    run_metadata = tf.RunMetadata()
                    samples = sess.run(fake_B, feed_dict={is_training: True},
                        options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                        run_metadata=run_metadata)
                else: 
                    samples = sess.run(fake_B, feed_dict={is_training: True})

                #  stats
                test_rate = 1000*(time.time()-starttime)
                print("[Test] [%2d] %.2f ms/step" % (idx, test_rate))
                if args.enable_profile:
                    summary_writer_test.add_run_metadata(run_metadata, 'global_step_%d'%idx)
                idx += 1

                ## save images
                samples = [utils.denormalize(img) for img in samples]
                samples = np.concatenate(samples, axis=1)
                save_dir = os.path.join('.', 'test')
                save_path = os.path.join(save_dir, 'test_%04d.png' % idx)
                # save_path = os.path.join(save_dir, '{}.png'.format(testset['name'][idx]))
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                utils.save_img(samples, save_path)

                if args.enable_profile:
                    with open(os.path.join(timeline_dir, 'timeline_%d.json' % idx), 'w') as f:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=True)
                        f.write(chrome_trace)

            except tf.errors.OutOfRangeError:
                break
