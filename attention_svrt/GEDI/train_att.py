import os, sys, time, re
sys.path.append('../') #puts model_depo on the path
import tensorflow as tf
import numpy as np
from exp_ops.data_loader import get_image_size, inputs
from exp_ops.helper_functions import make_dir, softmax_cost, fine_tune_prepare_layers, ft_non_optimized, class_accuracy
from gedi_config import GEDIconfig
from model_depo import attention_model_draw as att
from datetime import datetime
from argparse import ArgumentParser
from model_depo import vgg16_trainable as vgg16
def train_att(which_dataset,train_data=None,validation_data=None): #Fine tuning defaults to wipe out the final two FCs
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
        att_model = att.Attention()
       
        train_mode = tf.Variable(True, name='training')
        att_model.build(train_images,enc_size=config.enc_size,read_n=config.read_n,T=config.T,output_shape=config.output_shape,train_mode=train_mode)

        #Prepare the cost function
        cost = softmax_cost(att_model.fc2, train_labels)
        
        tf.scalar_summary("cost", cost)
        #print type(tf.trainable_variables()[0])
        print [x.name for x in tf.trainable_variables()]
        #Finetune the learning rates
        optimizer=tf.train.AdamOptimizer(learning_rate=config.new_lr,beta1=0.5)
        grads=optimizer.compute_gradients(cost)
        #v_i=att_model.v_i
	for i,(g,v) in enumerate(grads):
	    if g is not None:
            
		grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
	train_op=optimizer.apply_gradients(grads)
        #print a
        #Setup validation op
        eval_accuracy = class_accuracy(att_model.prob,train_labels) #training accuracy now...
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
            _,loss_value, acc= sess.run([train_op, cost, eval_accuracy])
            #print v_j
            losses.append(loss_value)
            accs.append(acc)
  
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                if step % 2000 == 0:
                    #Summaries
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                else:
                    #Training status
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value,
                                         config.train_batch / duration, float(duration)))
               
                if np.average(accs[-100:-1])>0.9 or np.average(losses[-100:-1])<0.1:
                     saver.save(sess, os.path.join(config.train_checkpoint, 'model_' + str(step) + '.ckpt'),
                 global_step=step)
                
                     coord.request_stop()

            # Save the model checkpoint periodically.
            if step % 1000 == 0 and step>0 :
                saver.save(sess, os.path.join(config.train_checkpoint, 'model_' + str(step) + '.ckpt'),
                 global_step=step)
                
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
   train_att('20')
