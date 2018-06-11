import os, sys, time, re
sys.path.append('../') #puts model_depo on the path
import tensorflow as tf
import numpy as np
from exp_ops.data_loader import get_image_size, inputs
from exp_ops.helper_functions import make_dir, find_ckpts, class_accuracy, plot_accuracies
from gedi_config import GEDIconfig
from model_depo import vgg16_trainable as vgg16
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
from model_depo import attention_model_draw as att
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
    print ckpts, ckpt_names
    #Prepare data on CPU
    with tf.device('/cpu:0'):
        val_images, val_labels = inputs(validation_data, config.validation_batch, im_shape, config.model_image_size[:2], num_epochs=1)

    #Prepare model on GPU
    with tf.device('/gpu:0'):
        att_model = att.Attention()
       
        validation_mode = tf.Variable(False, name='training')
        att_model.build(val_images,enc_size=config.enc_size,read_n=config.read_n,T=config.T,output_shape=config.output_shape,train_mode=validation_mode)
        image_0=val_images
        image_1=att_model.image_show
        image_loc=att_model.location
       # print image_0.get_shape()
       # print image_1[0].get_shape()
        #Setup validation op
        eval_accuracy = class_accuracy(att_model.prob,val_labels) #training accuracy now...

    #Set up saver
    saver = tf.train.Saver(tf.all_variables())

    #Loop through each checkpoint, loading the model weights, then testing the entire validation set
    ckpt_accs = []
    max_acc=0
    max_ind=0
    max_show_0=[]
    max_show_1=[]
    max_loc=[]
    for idx in tqdm(range(len(ckpts))):
        print ckpts[idx]
        accs = []
        show_0=np.array([])
        show_1=np.array([])
        show_loc=np.array([])
        try:
            
            #print type(show_0)
            #Initialize the graph
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))
            sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())) #need to initialize both if supplying num_epochs to inputs

            #Set up exemplar threading
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            saver.restore(sess, ckpts[idx])
            start_time = time.time()
            while not coord.should_stop():
                #print '1'
                #print type(accs)
                acc,aa,bb,cc=sess.run([eval_accuracy,image_0,image_1,image_loc])
                accs = np.append(accs,acc)
                if accs[-1]>0.8 and show_0.shape[-1]<5:
                    #print show_0.shape[-1]
                    #print aa.shape, bb
                    aa=aa
                    bb=bb
                    (x1,x2,x3,x4)=aa.shape
                    (y1,y2,y3,y4)=bb.shape
                    (z1,z2,z3)=cc.shape
                    aa=np.reshape(aa,(x1,x2,x3,x4,1))
                    bb=np.reshape(bb,(y1,y2,y3,y4,1)) 
                    cc=np.reshape(cc,(z1,z2,z3,1))
                    if show_0.shape[0]<=2:
                        #print sess.run([image_1])
                        
                        show_0=aa
                        show_1=bb
                        show_loc=cc
                    else:
                        #print sess.run([image_0])[0].shape, show_0.shape
                        
                        show_0=np.concatenate((show_0,aa),4)
		        show_1=np.concatenate((show_1,bb),4) 
                        show_loc=np.concatenate((show_loc,cc),3)   

        except tf.errors.OutOfRangeError:
            if np.mean(accs)>max_acc:
                max_acc=np.mean(accs)
                max_ind=idx
                max_show_0=show_0
                max_show_1=show_1
                max_loc=show_loc
            ckpt_accs.append(accs)
            print('Batch %d took %.1f seconds', idx, time.time() - start_time)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
   
           
    print ckpt_accs, ckpt_names

    #Plot everything
    plot_accuracies(ckpt_accs,ckpt_names,out_dir + 'validation_accuracies.png')
    np.savez(out_dir + 'validation_accuracies',ckpt_accs=ckpt_accs,ckpt_names=ckpt_names)
    np.savez(out_dir + 'att_verification_'+which_dataset,max_show_0=max_show_0,max_show_1=max_show_1,max_loc=max_loc)
    for idx in range(len(ckpts)):
        if idx!=max_ind:
            os.remove(ckpts[idx]+'.data-00000-of-00001')
            os.remove(ckpts[idx]+'.meta')
    
if __name__ == '__main__':
    test_vgg16('20')
