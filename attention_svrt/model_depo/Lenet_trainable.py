import os
import tensorflow as tf

import numpy as np 
import time
import inspect




class Lenet:
 

    def __init__(self, vgg16_npy_path='/home/xuefei/Project_spatial_reasoning/vgg16_svrt/pretrained_weights/Lenet.npy', trainable=False, fine_tune_layers=None):
        if vgg16_npy_path is not None:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            if fine_tune_layers is not None: #pop the specified keys from the weights that will be loaded
                for key in fine_tune_layers:
                    del self.data_dict[key]
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    def build(self, rgb, DO_SHARE=None,output_shape = None, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        if output_shape is None:
            output_shape = 10

        x=rgb
        
        self.conv1_1 = self.conv_layer(x, 1, 6, "conv1_1",DO_SHARE)
        #self.conv1_2 = self.conv_layer(self.conv1_1, 6, 6, "conv1_2",DO_SHARE)
        self.pool1 = self.max_pool(self.conv1_1, 'pool1')
      
        self.conv2_1 = self.conv_layer(self.pool1, 6, 16, "conv2_1",DO_SHARE)
        #self.conv2_2 = self.conv_layer(self.conv2_1, 12, 12, "conv2_2",DO_SHARE)
        self.pool2 = self.max_pool(self.conv2_1, 'pool2')

        self.fc11=self.fc_layer(self.pool2,28*28/16*16,120,'fc11',DO_SHARE)
        #self.fc12=self.fc_layer(self.fc11,120,N*N,'fc12',DO_SHARE)
        self.relu1 = tf.nn.relu(self.fc11)
        if train_mode is not None: #Consider changing these to numpy conditionals
            self.relu1 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu1, 0.5), lambda: self.relu1)
        elif self.trainable:
            self.relu1 = tf.nn.dropout(self.relu1, 0.5)

        self.fc12 = self.fc_layer(self.relu1, 120, output_shape, "fc12",DO_SHARE)

        self.prob = tf.nn.softmax(self.fc12, name="prob")

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name,DO_SHARE):
        with tf.variable_scope(name,reuse=DO_SHARE):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name,DO_SHARE):
        with tf.variable_scope(name,reuse=DO_SHARE):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
       # print self.data_dict.keys(),name
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
           # print 'da'
        else:

           # print 'de'
            value = initial_value

        if self.trainable:
            var = tf.get_variable(var_name,initializer=value) #get_variable, change the boolian to numpy
        else:
            var = tf.get_variable(var_name,initializer=tf.constant(value),trainable=False)

        self.var_dict[(name, idx)] = var

        print var_name, var.get_shape().as_list(), initial_value.get_shape()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="Lenet.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
