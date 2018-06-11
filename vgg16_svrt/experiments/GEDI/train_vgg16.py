import os, sys, time, re
sys.path.append('../../') #puts model_depo on the path
import tensorflow as tf
import numpy as np
from exp_ops.data_loader import get_image_size, inputs
from exp_ops.helper_functions import make_dir, softmax_cost, fine_tune_prepare_layers, ft_non_optimized, class_accuracy
from gedi_config import GEDIconfig
from model_depo import vgg16_trainable as vgg16
from datetime import datetime
from argparse import ArgumentParser

def train_vgg16(which_dataset,train_data=None,validation_data=None): #Fine tuning defaults to wipe out the final two FCs
    config = GEDIconfig(which_dataset)
    if train_data == None: #Use globals
        train_data = config.tfrecord_dir + 'train.tfrecords'
    if validation_data == None: #Use globals
        validation_data = config.tfrecord_dir + 'val.tfrecords'

    #Make output directories if they do not exist
    dt_stamp = re.split(' ',str(datetime.now()))[0].replace('-','_')
    config.train_checkpoint = os.path.join(config.train_checkpoint,config.which_dataset + '_' + dt_stamp) #timestamp this run
    out_dir = config.results + config.which_dataset + '/'
    dir_list = [config.train_checkpoint,config.train_summaries,config.results,out_dir]
    [make_dir(d) for d in dir_list]
    im_shape = get_image_size(config)

    #Prepare data on CPU
    with tf.device('/cpu:0'):
        train_images, train_labels = inputs(train_data, config.train_batch, im_shape, config.model_image_size[:2], train=config.data_augmentations, num_epochs=config.epochs)
        tf.image_summary('train images', train_images)

    #Prepare model on GPU
    with tf.device('/gpu:0'):
        vgg = vgg16.Vgg16(vgg16_npy_path=config.vgg16_weight_path,fine_tune_layers=config.fine_tune_layers)
        train_mode = tf.Variable(True, name='training')
        vgg.build(train_images,output_shape=config.output_shape,train_mode=train_mode)

        #Prepare the cost function
        cost = softmax_cost(vgg.fc8, train_labels)
        tf.scalar_summary("cost", cost)
        
        #Finetune the learning rates
        other_opt_vars,ft_opt_vars = fine_tune_prepare_layers(tf.trainable_variables(),config.fine_tune_layers) #for all variables in trainable variables, print name if there's duplicates you fucked up
        train_op = ft_non_optimized(cost,other_opt_vars,ft_opt_vars,tf.train.AdamOptimizer,config.hold_lr,config.new_lr) #actually is faster :)

        #Setup validation op
        eval_accuracy = class_accuracy(vgg.prob,train_labels) #training accuracy now...
        tf.scalar_summary("train accuracy", eval_accuracy)

    #Set up summaries and saver
    saver = tf.train.Saver(tf.all_variables(),max_to_keep=100)
    summary_op = tf.merge_all_summaries()

    #Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))
    sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())) #need to initialize both if supplying num_epochs to inputs
    summary_writer = tf.train.SummaryWriter(os.path.join(config.train_summaries,config.which_dataset + '_' + dt_stamp), sess.graph)

    #Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    #Start training loop
    try:
        step = 0
        losses = []
        accs=[]
        while not coord.should_stop():
            start_time = time.time()
            _, loss_value, acc = sess.run([train_op, cost, eval_accuracy])
            losses.append(loss_value)
            accs.append(acc)
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                if step % 500 == 0:
                    #Summaries
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                else:
                    #Training status
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value,
                                         config.train_batch / duration, float(duration)))

            # Save the model checkpoint periodically.
            if step % 500 == 0 and step>0 :
                saver.save(sess, os.path.join(config.train_checkpoint, 'model_' + str(step) + '.ckpt'),
                 global_step=step)
                if np.average(accs[-500:-1])>0.95:
                     coord.request_stop()
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (config.epochs, step))
        saver.save(sess, os.path.join(config.train_checkpoint, 'model_' + str(step) + '.ckpt'),
                 global_step=step)
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    np.save(out_dir + 'training_loss',losses)

if __name__ == '__main__':
   train_vgg16('1')
