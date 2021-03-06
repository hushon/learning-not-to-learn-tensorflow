import argparse
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
import model
import utils

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.global_step = 0
        self._timestamp = datetime.now().strftime('%Y%m%d%H%M')

        self._build_model()

        # checkpoint
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                            net=self.net,
                                            optimizer_color=self.optimizer_color,
                                            pred_net_r=self.pred_net_r,
                                            pred_net_g=self.pred_net_g,
                                            pred_net_b=self.pred_net_b,
                                            global_step=self.global_step,
                                            global_epoch=self.global_epoch)

        # summary writer
        self.summary_writer_train = tf.summary.create_file_writer(
            os.path.join(self.args.log_dir, "train"))
        self.summary_writer_val = tf.summary.create_file_writer(
            os.path.join(self.args.log_dir, "val"))

        self._build_dataset()

    def _build_model(self):
        # model
        self.net = model.convnet(self.args.dim_class)
        self.pred_net_r = model.Predictor(self.args.dim_bias)
        self.pred_net_g = model.Predictor(self.args.dim_bias)
        self.pred_net_b = model.Predictor(self.args.dim_bias)
        self.gradReverse = model.GradientReversalLayer()

        # loss function
        self.sparse_crossentropy = tf.keras.losses.SparseCategoricalCrossentropy()
        self.crossentropy = tf.keras.losses.CategoricalCrossentropy()

        # optimizer
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.args.lr, 40, 0.9)
        self.optimizer = tf.keras.optimizers.SGD(lr_schedule, 0.9)

        lr_schedule_color = tf.keras.optimizers.schedules.ExponentialDecay(self.args.lr, 40, 0.9)
        self.optimizer_color = tf.keras.optimizers.SGD(lr_schedule_color, 0.9)

        # metrics
        self.global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        self.global_epoch = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

        self.train_metrics = {"classifier_loss": tf.keras.metrics.Mean(name='classifier_loss'),
                            "color_loss": tf.keras.metrics.Mean(name='color_loss'),
                            "mi_loss": tf.keras.metrics.Mean(name='mi_loss'),
                            "classifier_accuracy": tf.keras.metrics.SparseCategoricalAccuracy(name='classifier_accuracy')}

        self.test_metrics = {"test_classifier_loss": tf.keras.metrics.Mean(name='test_classifier_loss'),
                            "test_color_loss": tf.keras.metrics.Mean(name='test_color_loss'),
                            "test_mi_loss": tf.keras.metrics.Mean(name='test_mi_loss'),
                            "test_classifier_accuracy": tf.keras.metrics.SparseCategoricalAccuracy(name='test_classifier_accuracy')}

    def _reset_train_metrics(self):
        for metric in self.train_metrics.values():
            metric.reset_states()

    def _reset_test_metrics(self):
        for metric in self.test_metrics.values():
            metric.reset_states()

    def _build_dataset(self):
        dataset = np.load(self.args.data_dir, allow_pickle=True, encoding='latin1').item()
        boundaries = list(range(0, 256, 256//self.args.dim_bias))[1:]
        self._quantize = lambda x: tf.raw_ops.Bucketize(input=x, boundaries=boundaries)

        self.train_ds = tf.data.Dataset.from_tensor_slices((dataset['train_image'], dataset['train_label']))\
                                    .map(self._preprocess).cache()\
                                    .shuffle(10000).batch(self.args.batch_size).prefetch(1)
        self.test_ds = tf.data.Dataset.from_tensor_slices((dataset['test_image'], dataset['test_label']))\
                                    .map(self._preprocess).cache()\
                                    .batch(self.args.batch_size).prefetch(1)

    def _preprocess(self, image, label):
        image = tf.cast(image, tf.float32)
        colormap = tf.image.resize(image, (14,14))
        bias = self._quantize(colormap)
        mask = tf.cast(tf.math.greater(colormap, 0.0), tf.float32)
        # r = self._quantize(colormap[..., 0])
        # g = self._quantize(colormap[..., 1])
        # b = self._quantize(colormap[..., 2])
        # bias = tf.stack([r, g, b], axis=-1)
        return (image/255.0, label, bias, mask)

    def _restore_checkpoint(self):
        try: self.checkpoint.restore(tf.train.latest_checkpoint(self.args.ckpt_dir)).expect_partial()
        except: print("Could not restore from checkpoint.")

    def _save_checkpoint(self):
        checkpoint_prefix = os.path.join(self.args.ckpt_dir, self._timestamp)
        self.checkpoint.save(file_prefix=checkpoint_prefix)

    @tf.function
    def _train_step(self, images, labels, bias, mask):
        with tf.GradientTape() as tape:
            feat_label, pred_label = self.net(images)

            _, pseudo_pred_r = self.pred_net_r(feat_label)
            _, pseudo_pred_g = self.pred_net_g(feat_label)
            _, pseudo_pred_b = self.pred_net_b(feat_label)

            loss_pred = self.sparse_crossentropy(labels, pred_label, sample_weight=mask)

            loss_pseudo_pred_r = self.crossentropy(pseudo_pred_r, pseudo_pred_r, sample_weight=mask)
            loss_pseudo_pred_g = self.crossentropy(pseudo_pred_g, pseudo_pred_g, sample_weight=mask)
            loss_pseudo_pred_b = self.crossentropy(pseudo_pred_b, pseudo_pred_b, sample_weight=mask)
            loss_pred_ps_color = (loss_pseudo_pred_r + loss_pseudo_pred_g + loss_pseudo_pred_b) / 3.

            loss = loss_pred + loss_pred_ps_color*self.args.loss_lambda

            gradients = tape.gradient(loss, self.net.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))

        self.train_metrics["classifier_loss"].update_state(loss_pred)
        self.train_metrics["mi_loss"].update_state(loss_pred_ps_color)
        self.train_metrics["classifier_accuracy"].update_state(labels, pred_label)

        with tf.GradientTape() as tape:
            feat_label, _ = self.net(images)

            color_label = self.gradReverse(feat_label)

            pred_r, _ = self.pred_net_r(color_label)
            pred_g, _ = self.pred_net_g(color_label)
            pred_b, _ = self.pred_net_b(color_label)

            loss_pred_r = self.sparse_crossentropy(bias[..., 0], pred_r, sample_weight=mask)
            loss_pred_g = self.sparse_crossentropy(bias[..., 1], pred_g, sample_weight=mask)
            loss_pred_b = self.sparse_crossentropy(bias[..., 2], pred_b, sample_weight=mask)
            loss_pred_color = loss_pred_r + loss_pred_g + loss_pred_b

            # TODO: optimizer must update feat_label part of self.net
            gradients = tape.gradient(loss_pred_color, self.net.trainable_variables\
                + self.pred_net_r.trainable_variables\
                + self.pred_net_g.trainable_variables\
                + self.pred_net_b.trainable_variables)
            self.optimizer_color.apply_gradients(zip(gradients, self.net.trainable_variables\
                + self.pred_net_r.trainable_variables\
                + self.pred_net_g.trainable_variables\
                + self.pred_net_b.trainable_variables))

        self.train_metrics["color_loss"].update_state(loss_pred_color)
        self.global_step = self.global_step.assign_add(1)

    @tf.function
    def _train_step_baseline(self, images, labels, bias, mask):
        with tf.GradientTape() as tape:
            _, pred_label = self.net(images)
            loss_pred = self.sparse_crossentropy(labels, pred_label, sample_weight=mask)
            gradients = tape.gradient(loss_pred, self.net.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))

        self.train_metrics["classifier_loss"].update_state(loss_pred)
        self.train_metrics["classifier_accuracy"].update_state(labels, pred_label)
        self.global_step = self.global_step.assign_add(1)

    @tf.function
    def _test_step(self, images, labels, bias, mask):
        feat_label, pred_label = self.net(images)

        pred_r, pseudo_pred_r = self.pred_net_r(feat_label)
        pred_g, pseudo_pred_g = self.pred_net_g(feat_label)
        pred_b, pseudo_pred_b = self.pred_net_b(feat_label)

        loss_pred = self.sparse_crossentropy(labels, pred_label, sample_weight=mask)

        loss_pseudo_pred_r = self.crossentropy(pseudo_pred_r, pseudo_pred_r, sample_weight=mask)
        loss_pseudo_pred_g = self.crossentropy(pseudo_pred_g, pseudo_pred_g, sample_weight=mask)
        loss_pseudo_pred_b = self.crossentropy(pseudo_pred_b, pseudo_pred_b, sample_weight=mask)
        loss_pred_ps_color = (loss_pseudo_pred_r + loss_pseudo_pred_g + loss_pseudo_pred_b) / 3.

        loss_pred_r = self.sparse_crossentropy(bias[..., 0], pred_r, sample_weight=mask)
        loss_pred_g = self.sparse_crossentropy(bias[..., 1], pred_g, sample_weight=mask)
        loss_pred_b = self.sparse_crossentropy(bias[..., 2], pred_b, sample_weight=mask)
        loss_pred_color = loss_pred_r + loss_pred_g + loss_pred_b

        self.test_metrics["test_classifier_loss"].update_state(loss_pred)
        self.test_metrics["test_mi_loss"].update_state(loss_pred_ps_color)
        self.test_metrics["test_color_loss"].update_state(loss_pred_color)
        self.test_metrics["test_classifier_accuracy"].update_state(labels, pred_label)

    @tf.function
    def _write_summary_train(self, step):
        with self.summary_writer_train.as_default():
            with tf.name_scope("loss"):
                tf.summary.scalar("classifier_loss", self.train_metrics["classifier_loss"].result(), step=step)
                tf.summary.scalar("color_loss", self.train_metrics["color_loss"].result(), step=step)
                tf.summary.scalar("mi_loss", self.train_metrics["mi_loss"].result(), step=step)
            with tf.name_scope("accuracy"):
                tf.summary.scalar("classifier_accuracy", self.train_metrics["classifier_accuracy"].result()*100, step=step)

    @tf.function
    def _write_summary_val(self, step):
        with self.summary_writer_val.as_default():
            with tf.name_scope("loss"):
                tf.summary.scalar("classifier_loss", self.test_metrics["test_classifier_loss"].result(), step=step)
                tf.summary.scalar("color_loss", self.test_metrics["test_color_loss"].result(), step=step)
                tf.summary.scalar("mi_loss", self.test_metrics["test_mi_loss"].result(), step=step)
            with tf.name_scope("accuracy"):
                tf.summary.scalar("classifier_accuracy", self.test_metrics["test_classifier_accuracy"].result()*100.0, step=step)


    def train(self):
        # restore checkpoint
        self._restore_checkpoint()

        # train
        if self.args.train_baseline:
            __train_step = self._train_step_baseline
        else:
            __train_step = self._train_step

        for epoch in range(self.global_epoch.value(), self.args.max_epoch):
            for batch in self.train_ds:
                self._reset_train_metrics()
                __train_step(*batch)
                self._write_summary_train(self.global_step)

            self.global_epoch.assign_add(1)

            print(f"Epoch: {epoch}, "
                f"Loss: {self.train_metrics['classifier_loss'].result():.4f}, "
                f"Acc: {self.train_metrics['classifier_accuracy'].result()*100:.4f}")

            # validation
            if epoch % 5 == 0:
                self._reset_test_metrics()
                for batch in self.test_ds:
                    self._test_step(*batch)
                self._write_summary_val(self.global_step)

                print(f"Test Loss: {self.test_metrics['test_classifier_loss'].result():.4f}, "
                    f"Test Acc: {self.test_metrics['test_classifier_accuracy'].result()*100.0:.4f}")

        # save checkpoint
        self._save_checkpoint()

        print("Finished job.")

    def test(self):
        # restore checkpoint
        self._restore_checkpoint()

        self._reset_test_metrics()
        for batch in self.test_ds:
            self._test_step(*batch)

        print(f"Avg Test Acc: {self.test_metrics['test_classifier_accuracy'].result()*100:.4f}")

        print("Finished job.")

def main():
    # parse options
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--phase', dest='phase', default='train', help='train, test')
    parser.add_argument('--train_baseline', dest='train_baseline', action='store_true', help='train baseline model')
    parser.add_argument('--data_dir', dest='data_dir', default=None, help='dataset dir')
    parser.add_argument('--log_dir', dest='log_dir', default='./logs/', help='log dir')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./checkpoint/', help='checkpoint dir')
    parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=100, help='maximum epoch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--image_size', dest='image_size', type=int, default=28, help='pixel size')
    parser.add_argument('--input_c_dim', dest='input_c_dim', type=int, default=3, help='number of channels')
    parser.add_argument('--dim_class', dest='dim_class', type=int, default=10, help='number of class categories')
    parser.add_argument('--dim_bias', dest='dim_bias', type=int, default=16, help='bias dimension')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--loss_lambda', dest='loss_lambda', type=float, default=0.01, help='lambda coeff')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0005, help='l2 weight decay')
    args = parser.parse_args()
    if not args.data_dir:
        args.data_dir = utils.ask_openfile(("numpy files","*.npy"))
    for k, v in vars(args).items(): print(f'{k} = {v}')

    # Enable GPU memory growth option
    # needed on GeForce RTX GPUs as workaround to a bug in tensorflow
    # Reference: https://github.com/tensorflow/tensorflow/issues/24828
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    # Execute training or testing
    trainer = Trainer(args)
    if args.phase == 'train':
        trainer.train()
    elif args.phase == 'test':
        trainer.test()
    else:
        raise ValueError

if __name__ == '__main__':
    main()
