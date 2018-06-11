import os, sys, time, re
sys.path.append('../../') #puts model_depo on the path
import tensorflow as tf
import numpy as np
from exp_ops.data_loader import get_image_size, inputs
from exp_ops.helper_functions import make_dir, find_ckpts, class_accuracy, plot_accuracies
from gedi_config import GEDIconfig
from model_depo import vgg16_trainable as vgg16
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm

def test_vgg16(which_dataset,validation_data=None,model_dir=None): #Fine tuning defaults to wipe out the final two FCs
    config = GEDIconfig(which_dataset)
    if validation_data == None: #Use globals
        validation_data = config.tfrecord_dir + 'val.tfrecords'

    #Make output directories if they do not exist
    out_dir = config.results + config.which_dataset + '/'
    dir_list = [config.results,out_dir]
    [make_dir(d) for d in dir_list]
    im_shape = get_image_size(config)

    #Find model checkpoints
    ckpts, ckpt_names = find_ckpts(config)

    #Prepare data on CPU
    with tf.device('/cpu:0'):
        val_images, val_labels = inputs(validation_data, config.validation_batch, im_shape, config.model_image_size[:2], num_epochs=1)

    #Prepare model on GPU
    with tf.device('/gpu:0'):
        vgg = vgg16.Vgg16(vgg16_npy_path=config.vgg16_weight_path,fine_tune_layers=config.fine_tune_layers)
        validation_mode = tf.Variable(False, name='training')
        vgg.build(val_images,output_shape=config.output_shape,train_mode=validation_mode)

        #Setup validation op
        eval_accuracy = class_accuracy(vgg.prob,val_labels) #training accuracy now...

    #Set up saver
    saver = tf.train.Saver(tf.all_variables())

    #Loop through each checkpoint, loading the model weights, then testing the entire validation set
    ckpt_accs = []
    for idx in tqdm(range(len(ckpts))):
        accs = []
        try:
            #Initialize the graph
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))
            sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())) #need to initialize both if supplying num_epochs to inputs

            #Set up exemplar threading
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            saver.restore(sess, ckpts[idx])
            start_time = time.time()
            while not coord.should_stop():
                accs = np.append(accs,sess.run([eval_accuracy]))

        except tf.errors.OutOfRangeError:
            ckpt_accs.append(accs)
            print('Batch %d took %.1f seconds', idx, time.time() - start_time)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

    #Plot everything
    plot_accuracies(ckpt_accs,ckpt_names,out_dir + 'validation_accuracies.png')
    np.savez(out_dir + 'validation_accuracies',ckpt_accs=ckpt_accs,ckpt_names=ckpt_names)

if __name__ == '__main__':
    test_vgg16('1')
