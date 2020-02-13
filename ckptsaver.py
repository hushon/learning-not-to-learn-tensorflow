import os
import tensorflow as tf

def save_checkpoint(sess, saver, save_dir, prefix, global_step):
    '''Saves session to .ckpt file.
    Args:
        sess: tf.Session object
        saver: tf.train.Saver object
        save_dir (str): checkpoint directory
        prefix (str): naming prefix for .ckpt file
        global_step (int): # iteration of training step'''
    ckpt_dir = os.path.normpath(save_dir)
    ckpt_path = saver.save(sess, save_path=os.path.join(save_dir, prefix), global_step=global_step)
    print("[*] Saved session to checkpoint: %s" % os.path.basename(ckpt_path))

def load_checkpoint(sess, saver, load_dir):
    '''Restores session from .ckpt file. 
    Function raises tf.errors.NotFoundError under failure.
    Args:
        sess: tf.Session object
        saver: tf.train.Saver object
        load_dir (str): checkpoint directory'''
    load_dir = os.path.normpath(load_dir)
    ckpt_state = tf.train.get_checkpoint_state(load_dir)
    if ckpt_state and ckpt_state.model_checkpoint_path:
        saver.restore(sess, save_path=ckpt_state.model_checkpoint_path)
        print("[*] Restored session from checkpoint: %s" % os.path.basename(ckpt_state.model_checkpoint_path))
    else:
        raise tf.errors.NotFoundError(None, None, "[!] Unable to restore from checkpoint: %s" % load_dir)