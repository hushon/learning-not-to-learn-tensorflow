import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm, trange
from PIL import Image

import network
import ckptsaver
import utils
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
                                     shape=[None,
                                            args.image_size,
                                            args.image_size,
                                            args.input_c_dim],
                                     name='input_image')
        label_class = tf.placeholder(tf.float32,
                                     shape=[None, args.dim_class],
                                     name='label_class')
        label_bias = tf.placeholder(tf.float32,
                                    shape=[None, 3, 14, 14, args.dim_bias],
                                    name='label_bias')
        # label_bias = tf.placeholder(tf.float32,
        #                             shape=[None, 3, args.dim_bias],
        #                             name='label_bias')
        is_training = tf.placeholder(tf.bool,
                                     name='is_training')
        loss_lambda = tf.placeholder(tf.float32,
                                     name='loss_lambda')        

        feature = network.f(input_image,
                            output_dim=32,
                            is_training=is_training,
                            name='feature_extractor')
        output_class, output_class_logits = network.g(feature,
                                                      output_dim=args.dim_class,
                                                      is_training=is_training,
                                                      name='class_predictor')
        output_bias_r, output_bias_r_logits = network.h(feature,
                                                    output_dim=args.dim_bias,
                                                    is_training=is_training,
                                                    name='bias_r_predictor')
        output_bias_g, output_bias_g_logits = network.h(feature,
                                                    output_dim=args.dim_bias,
                                                    is_training=is_training,
                                                    name='bias_g_predictor')
        output_bias_b, output_bias_b_logits = network.h(feature,
                                                    output_dim=args.dim_bias,
                                                    is_training=is_training,
                                                    name='bias_b_predictor')

        global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=tf.constant(0), trainable=False)
        global_epoch = tf.get_variable('global_epoch', dtype=tf.int32, initializer=tf.constant(0), trainable=False)

        '''loss'''
        with tf.variable_scope("loss_classifier"):
            loss_classifier = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label_class,
                logits=output_class_logits))

        with tf.variable_scope("loss_mi"):
            loss_mi = (tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(
                                                labels=output_bias_r,
                                                logits=output_bias_r_logits))
                        + tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(
                                                labels=output_bias_g,
                                                logits=output_bias_g_logits))
                        + tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(
                                                labels=output_bias_b,
                                                logits=output_bias_b_logits))) / 3.0

        with tf.variable_scope("loss_bias"):
            loss_bias_r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label_bias[:, 0],
                logits=output_bias_r_logits))
            loss_bias_g = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label_bias[:, 1],
                logits=output_bias_g_logits))
            loss_bias_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label_bias[:, 2],
                logits=output_bias_b_logits))
            loss_bias = loss_bias_r + loss_bias_g + loss_bias_b

        with tf.variable_scope("l2_weight_decay"):
            loss_l2_reg = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        '''accuracy'''
        with tf.variable_scope("acc_classifier"):
            acc_classifier = tf.reduce_mean(tf.cast(tf.equal(
                tf.argmax(output_class, axis=-1),
                tf.argmax(label_class, axis=-1)), tf.float32))
        
        with tf.variable_scope("acc_bias"):
            acc_bias_r = tf.reduce_mean(tf.cast(tf.equal(
                tf.argmax(output_bias_r, axis=-1),
                tf.argmax(label_bias[:, 0], axis=-1)), tf.float32))
            acc_bias_g = tf.reduce_mean(tf.cast(tf.equal(
                tf.argmax(output_bias_g, axis=-1),
                tf.argmax(label_bias[:, 1], axis=-1)), tf.float32))
            acc_bias_b = tf.reduce_mean(tf.cast(tf.equal(
                tf.argmax(output_bias_b, axis=-1),
                tf.argmax(label_bias[:, 2], axis=-1)), tf.float32))
            acc_bias = (acc_bias_r + acc_bias_g + acc_bias_b) / 3

        '''optimizer'''
        f_vars = [var for var in tf.trainable_variables() if 'feature_extractor' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'class_predictor' in var.name]
        h_vars = [var for var in tf.trainable_variables() if 'bias_r_predictor' in var.name] +\
                 [var for var in tf.trainable_variables() if 'bias_g_predictor' in var.name] +\
                 [var for var in tf.trainable_variables() if 'bias_b_predictor' in var.name]
        
        assert len(f_vars) > 0 and len(g_vars) > 0 and len(h_vars) > 0

        optimizer1 = tf.train.AdamOptimizer(args.lr)
        optimizer2 = tf.train.AdamOptimizer(args.lr)
        # decaying_lr = tf.train.exponential_decay(args.lr,
        #                 global_step=global_step,
        #                 decay_steps=40,
        #                 decay_rate=0.9)
        # optimizer1 = tf.train.MomentumOptimizer(decaying_lr, momentum=0.9)
        # optimizer2 = tf.train.MomentumOptimizer(decaying_lr, momentum=0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op_classifier = optimizer1.minimize(loss_classifier + loss_lambda*loss_mi + args.weight_decay*loss_l2_reg, var_list=g_vars + f_vars)
            # gradients = optimizer2.compute_gradients(loss_bias + args.weight_decay*loss_l2_reg, var_list=f_vars + h_vars)
            # gradients = [(-grad, var) if 'feature_extractor' in var.name else (grad, var) for (grad, var) in gradients]
            # self.train_op_bias = optimizer2.apply_gradients(gradients)
            # self.train_op_classifier = optimizer1.minimize(loss_classifier + loss_lambda*loss_mi - loss_lambda*loss_bias + args.weight_decay*loss_l2_reg, var_list=g_vars + f_vars)
            self.train_op_bias = optimizer2.minimize(loss_bias + args.weight_decay*loss_l2_reg, var_list=h_vars)

        increment_global_step = tf.assign_add(global_step, 1)
        increment_global_epoch = tf.assign_add(global_epoch, 1)

        '''Define summary'''
        input_image_summary = tf.summary.image("input_image_summary", input_image)

        with tf.variable_scope('loss_functions'):
            loss_classifier_summary = tf.summary.scalar("loss_classifier_summary", loss_classifier)
            loss_bias_summary = tf.summary.scalar("loss_mi_summary", loss_mi)
            loss_bias_summary = tf.summary.scalar("loss_bias_summary", loss_bias)

        with tf.variable_scope('accuracy'):
            acc_classifier_summary = tf.summary.scalar("acc_classifier_summary", acc_classifier)
            acc_bias_summary = tf.summary.scalar("acc_bias_summary", acc_bias)

        with tf.variable_scope("f_weights_summary"):
            f_weights_summary = tf.summary.merge([tf.summary.histogram(var.name, var) for var in f_vars])
        with tf.variable_scope("g_weights_summary"):
            g_weights_summary = tf.summary.merge([tf.summary.histogram(var.name, var) for var in g_vars])
        with tf.variable_scope("h_weights_summary"):
            h_weights_summary = tf.summary.merge([tf.summary.histogram(var.name, var) for var in h_vars])
        summary_op = tf.summary.merge_all()

        self.input_image = input_image
        self.label_class = label_class
        self.label_bias = label_bias
        self.is_training = is_training
        self.loss_lambda = loss_lambda

        self.output_class = output_class
        self.feature = feature

        self.summary_op = summary_op

        self.global_step = global_step
        self.global_epoch = global_epoch
        self.increment_global_step = increment_global_step
        self.increment_global_epoch = increment_global_epoch

    def train(self):
        """Train pix2pix"""
        sess = self.sess
        args = self.args

        # saver
        saver = tf.train.Saver(max_to_keep=1)

        # summary writer
        summary_writer = tf.summary.FileWriter(
            os.path.join(os.path.normpath(args.log_dir), 'train'),
            sess.graph)

        summary_writer_val = tf.summary.FileWriter(
            os.path.join(os.path.normpath(args.log_dir), 'val'),
            sess.graph)

        ## restore checkpoint or initialize.
        load_dir = os.path.normpath(args.ckpt_dir)
        try: 
            ckptsaver.load_checkpoint(sess, saver, load_dir)
        except tf.errors.NotFoundError: 
            sess.run(tf.global_variables_initializer())
            print(" [*] initialized variables")

        ## load datasets
        dataset = np.load(args.data_dir, allow_pickle=True, encoding='latin1').item()

        trainset = {k: v for k, v in dataset.items() if 'train' in k}
        testset = {k: v for k, v in dataset.items() if 'test' in k}

        ## preprocess dataset
        train_bias = []
        for x in trainset['train_image']:
            x = np.array(Image.fromarray(x).resize((14,14)))
            r = np.eye(args.dim_bias)[utils.quantize(x[..., 0], args.dim_bias)]
            g = np.eye(args.dim_bias)[utils.quantize(x[..., 1], args.dim_bias)]
            b = np.eye(args.dim_bias)[utils.quantize(x[..., 2], args.dim_bias)]
            train_bias.append([r,g,b])
        train_bias = np.array(train_bias)
        trainset.update({'train_bias': train_bias})

        # train_bias = []
        # for x in trainset['train_image']:
        #     r = np.eye(args.dim_bias)[utils.quantize(x[..., 0].max())]
        #     g = np.eye(args.dim_bias)[utils.quantize(x[..., 1].max())]
        #     b = np.eye(args.dim_bias)[utils.quantize(x[..., 2].max())]
        #     train_bias.append([r,g,b])
        # train_bias = np.array(train_bias)
        # trainset.update({'train_bias': train_bias})


        train_label = []
        for x in trainset['train_label']:
            train_label.append(np.eye(args.dim_class)[x])
        train_label = np.array(train_label)
        trainset.update({'train_label': train_label})

        test_bias = []
        for x in testset['test_image']:
            x = np.array(Image.fromarray(x).resize((14,14)))
            r = np.eye(args.dim_bias)[utils.quantize(x[..., 0], args.dim_bias)]
            g = np.eye(args.dim_bias)[utils.quantize(x[..., 1], args.dim_bias)]
            b = np.eye(args.dim_bias)[utils.quantize(x[..., 2], args.dim_bias)]
            test_bias.append([r,g,b])
        test_bias = np.array(test_bias)
        testset.update({'test_bias': test_bias})

        # test_bias = []
        # for x in testset['test_image']:
        #     r = np.eye(args.dim_bias)[utils.quantize(x[..., 0].max())]
        #     g = np.eye(args.dim_bias)[utils.quantize(x[..., 1].max())]
        #     b = np.eye(args.dim_bias)[utils.quantize(x[..., 2].max())]
        #     test_bias.append([r,g,b])
        # test_bias = np.array(test_bias)
        # testset.update({'test_bias': test_bias})


        test_label = []
        for x in testset['test_label']:
            test_label.append(np.eye(args.dim_class)[x])
        test_label = np.array(test_label)
        testset.update({'test_label': test_label})

        trainset = dataloader.Dataloader().from_dict(trainset)
        testset = dataloader.Dataloader().from_dict(testset)

        ## training loop
        for _ in trange(self.global_epoch.eval(), args.max_epoch):

            for batch in trainset.shuffle(None).iter(args.batch_size, False):

                if args.train_baseline:
                    feed_dict = {self.input_image: batch['train_image'],
                                self.label_class: batch['train_label'],
                                self.label_bias: batch['train_bias'],
                                self.loss_lambda: 0.0,
                                self.is_training: True}

                    _ = sess.run(self.train_op_classifier, feed_dict=feed_dict)
                    summary_str = sess.run(self.summary_op, feed_dict=feed_dict)

                else:
                    feed_dict = {self.input_image: batch['train_image'],
                                self.label_class: batch['train_label'],
                                self.label_bias: batch['train_bias'],
                                self.loss_lambda: args.loss_lambda,
                                self.is_training: True}

                    _ = sess.run(self.train_op_classifier, feed_dict=feed_dict)
                    _ = sess.run(self.train_op_bias, feed_dict=feed_dict)
                    summary_str = sess.run(self.summary_op, feed_dict=feed_dict)

                # write summary
                summary_writer.add_summary(summary_str, self.global_step.eval())

                # increment global step
                sess.run(self.increment_global_step)

            # validation summary
            if self.global_epoch.eval() % 5 == 0:
                batch = testset.shuffle(None).head(args.batch_size)
                feed_dict = {self.input_image: batch['test_image'],
                             self.label_class: batch['test_label'],
                             self.label_bias: batch['test_bias'],
                             self.loss_lambda: args.loss_lambda,
                             self.is_training: True}
                summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
                summary_writer_val.add_summary(summary_str, self.global_step.eval())

            # increment global epoch
            sess.run(self.increment_global_epoch)

        # save checkpoint
        ckptsaver.save_checkpoint(sess, saver, args.ckpt_dir, 'model', self.global_epoch)

    def test(self):
        args = self.args
        sess = self.sess

        ## restore checkpoint or initialize.
        saver = tf.train.Saver(max_to_keep=1)
        load_dir = os.path.normpath(args.ckpt_dir)
        try: 
            ckptsaver.load_checkpoint(sess, saver, load_dir)
        except tf.errors.NotFoundError: 
            sess.run(tf.global_variables_initializer())
            print(" [*] initialized variables")

        ## load dataset
        dataset = np.load(args.data_dir, allow_pickle=True, encoding='latin1')
        dataset = dataset.item()

        ## preprocess dataset
        dataset = {k: v for k, v in dataset.items() if 'test' in k}
        testset = dataloader.Dataloader().from_dict(dataset)

        # run test
        pred_data = {"accuracy": [],
                     "result": [],
                     "feature": [],
                     "label": dataset["test_label"]}

        for batch in tqdm(testset.iter(args.batch_size, False)):

            feed_dict = {self.input_image: batch['test_image'],
                         self.is_training: False}

            # prediction accuracy
            output = sess.run(self.output_class, feed_dict=feed_dict)
            output_feature = sess.run(self.feature, feed_dict=feed_dict)

            pred_data["accuracy"].append(np.argmax(output, axis=-1) == batch['test_label'])
            pred_data["result"].append(output)
            pred_data["feature"].append(output_feature)

        pred_data["result"] = np.concatenate(pred_data["result"], axis=0)
        pred_data["feature"] = np.concatenate(pred_data["feature"], axis=0)
        print('test acc: {}'.format(np.mean(pred_data["accuracy"])))

        # save data
        np.save('./pred_data.npy', pred_data)
