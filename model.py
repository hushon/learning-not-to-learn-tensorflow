import os
from tkinter import Tk, filedialog
import time

import tensorflow as tf
import numpy as np
from tqdm import tqdm, trange

import network
import ckptsaver

import dataloader

class LNTL:
    def __init__(self, sess, args):

        self.sess = sess
        self.args = args

        self._build_model()

    def _build_model(self):
        '''Defines TensorFlow graph.'''

        args = self.args

        print(' [*] Building tensorflow graph')

        '''main graph'''
        input_image = tf.placeholder(tf.float32,
                                     shape=[None, args.image_size, args.image_size, args.input_c_dim],
                                     name='input_image')
        label_class = tf.placeholder(tf.float32,
                                     shape=[None, args.dim_class],
                                     name='label_class')
        label_bias = tf.placeholder(tf.float32,
                                    shape=[None, args.dim_bias],
                                    name='label_bias')
        is_training = tf.placeholder(tf.bool,
                                     name='is_training')

        feature = network.f(input_image,
                            output_dim=32,
                            is_training=is_training,
                            name='feature_extractor')
        output_class, output_class_logits = network.g(feature,
                                                      output_dim=args.dim_class,
                                                      is_training=is_training,
                                                      name='class_predictor')
        output_bias, output_bias_logits = network.h(feature,
                                                    output_dim=args.dim_bias,
                                                    is_training=is_training,
                                                    name='bias_predictor')

        '''loss'''
        with tf.variable_scope("loss_functions"):
            loss_classifier = tf.nn.softmax_cross_entropy_with_logits(
                labels=label_class,
                logits=output_class_logits)
            loss_bias = tf.reduce_mean(tf.abs(label_bias - output_bias))
            loss = loss_classifier - args.loss_lambda*loss_bias

        '''optimizer'''
        f_vars = [var for var in tf.trainable_variables() if 'feature_extractor' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'class_predictor' in var.name]
        h_vars = [var for var in tf.trainable_variables() if 'bias_predictor' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op_classifier = tf.train.AdamOptimizer(args.lr).minimize(loss, var_list=f_vars + g_vars)
            train_op_bias = tf.train.AdamOptimizer(args.lr).minimize(loss_bias, var_list=h_vars)

        global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=tf.constant(0), trainable=False)
        global_epoch = tf.get_variable('global_epoch', dtype=tf.int32, initializer=tf.constant(0), trainable=False)
        increment_global_step = tf.assign_add(global_step, 1)
        increment_global_epoch = tf.assign_add(global_epoch, 1)

        '''Define summary'''
        input_image_summary = tf.summary.image("input_image", input_image)

        loss_classifier_summary = tf.summary.scalar("loss_classifier", tf.reduce_mean(loss_classifier))
        loss_bias_summary = tf.summary.scalar("loss_bias", tf.reduce_mean(loss_bias))
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
        self.is_training = is_training

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

        is_training = self.is_training

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
        dataset = {k: v for k, v in dataset.items() if 'train' in k}

        dataset['train_image'] = dataset['train_image'] / 127.5 - 1.0

        train_bias = []
        for x in dataset['train_image']:
            r = x[..., 0].max() / 127.5 - 1.0
            g = x[..., 1].max() / 127.5 - 1.0
            b = x[..., 2].max() / 127.5 - 1.0
            train_bias.append([r,g,b])
        train_bias = np.array(train_bias)
        dataset.update({'train_bias': train_bias})

        train_label = []
        for x in dataset['train_label']:
            train_label.append(np.eye(args.dim_class)[x])
        train_label = np.array(train_label)
        dataset.update({'train_label': train_label})

        dataset = dataloader.Dataloader().from_dict(dataset)

        ## training loop
        for _ in trange(self.global_epoch.eval(), args.max_epoch):

            for batch in dataset.shuffle(None).iter(args.batch_size, False):

                feed_dict = {self.input_image: batch['train_image'],
                             self.label_class: batch['train_label'],
                             self.label_bias: batch['train_bias'],
                             self.is_training: True}

                # train step
                _ = sess.run(self.train_op_classifier, feed_dict=feed_dict)
                _ = sess.run(self.train_op_bias, feed_dict=feed_dict)
                summary_str = sess.run(self.summary_op, feed_dict=feed_dict)

                # write summary
                summary_writer.add_summary(summary_str, self.global_step.eval())

                # increment global step
                sess.run(self.increment_global_step)

            # increment global epoch
            sess.run(self.increment_global_epoch)

        # save checkpoint
        ckptsaver.save_checkpoint(sess, saver, args.ckpt_dir, 'model', self.global_step)

    def test(self):
        args = self.args
        sess = self.sess

        is_training = self.is_training

        ## restore checkpoint or initialize.
        saver = tf.train.Saver(max_to_keep=1)
        load_dir = os.path.normpath(args.ckpt_dir)
        try: 
            ckptsaver.load_checkpoint(sess, saver, load_dir)
        except tf.errors.NotFoundError: 
            sess.run(tf.global_variables_initializer())
            print(" [*] initialized variables")

        ## load dataset
        root = Tk()
        filepath = filedialog.askopenfilename(title='Select file', 
                                              filetypes = (("numpy files","*.npy"),("all files","*.*")))
        filepath = os.path.normpath(filepath)
        root.withdraw()

        dataset = np.load(filepath, allow_pickle=True, encoding='latin1')
        dataset = dataset.item()

        ## preprocess dataset
        dataset = {k: v for k, v in dataset.items() if 'test' in k}
        dataset['test_image'] = dataset['test_image'] / 127.5 - 1.0
        dataset = dataloader.Dataloader().from_dict(dataset)

        # run test
        accuracy = []
        for batch in tqdm(dataset.iter(args.batch_size, False)):

            feed_dict = {self.input_image: batch['test_image'],
                         self.label_class: batch['test_label'],
                         self.is_training: False}

            # prediction
            output = sess.run(self.output_class, feed_dict=feed_dict)
            output = np.argmax(output, axis=-1)

            # accuracy
            accuracy.append(output == np.argmax(batch['test_label'], axis=-1))

        print('test acc: {}'.format(np.mean(accuracy)))