import os, sys, time, re
sys.path.append('../') #puts model_depo on the path
import tensorflow as tf
import numpy as np
from exp_ops.helper_functions import softmax_cost, class_accuracy
from datetime import datetime
from six.moves import cPickle as pickle
from six.moves import range
from model_depo import Lenet_trainable as lenet
def train_lenet(): #Fine tuning defaults to wipe out the final two FCs
    pickle_file = 'notMNIST.pickle'
   
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      train_dataset = save['train_dataset']
      train_data_labels = save['train_labels']
      valid_dataset = save['valid_dataset']
      valid_labels = save['valid_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']
      del save  # hint to help gc free up memory\n",
      print('Training set', train_dataset.shape, train_data_labels.shape)
      print('Validation set', valid_dataset.shape, valid_labels.shape)
      print('Test set', test_dataset.shape, test_labels.shape)
    image_size = 28
    num_labels = 10
    num_channels = 1 
    '''def accuracy(predictions, labels):
      print predictions.shape
      print labels.shape
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])'''
    def reformat(dataset, labels):
      dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
      #labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
      return dataset, labels
    train_dataset, train_data_labels = reformat(train_dataset, train_data_labels)
    #"valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)\n",
    #"test_dataset, test_labels = reformat(test_dataset, test_labels)\n",
    print('Training set', train_dataset.shape, train_data_labels.shape)
    #"print('Validation set', valid_dataset.shape, valid_labels.shape)\n",
    #"print('Test set', test_dataset.shape, test_labels.shape)"
    train_images = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
    train_labels = tf.placeholder(tf.int32, shape=(None,))
      
    Le = lenet.Lenet()
    train_mode = tf.Variable(True, name='training')
    Le.build(train_images,train_mode=train_mode)

#Prepare the cost function
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Le.fc12,train_labels))
    cost=softmax_cost(Le.fc12,train_labels) 
   # print Le.prob.get_shape() 
    eval_accuracy = class_accuracy(Le.prob,train_labels) #training accuracy now...
    global_step=tf.Variable(0)
    learning_rate=tf.train.exponential_decay(0.05,global_step,100,0.95)
    optimizer=tf.train.AdamOptimizer(0.0001).minimize(cost)

    #Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))
    sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())) #need to initialize both if supplying num_epochs to inputs
   
    #Start training loop
    batch_size=50
    step = 0
    losses = []
    accs=[]
    while step<40000:
      offset = (step * batch_size) % (train_data_labels.shape[0] - batch_size)
      batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
      batch_labels = train_data_labels[offset:(offset + batch_size)]
      feed_dict = {train_images : batch_data, train_labels : batch_labels}   
      _, loss_value, acc = sess.run([optimizer, cost, eval_accuracy],feed_dict=feed_dict)
      losses.append(loss_value)
      accs.append(acc)
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
      if step % 100 == 0:
        print (step,loss_value,acc)
		
      step += 1
      if step%5000==0 and step>1:
        Le.save_npy(sess,"Lenet_save.npy")

  

if __name__ == '__main__':
   train_lenet()
