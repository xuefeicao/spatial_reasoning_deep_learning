import os,sys
import tensorflow as tf
import numpy as np
import time
import inspect
import vgg16_trainable as vgg16
A=128
B=128
eps=1e-8
VGG_MEAN = [103.939, 116.779, 123.68]
DO_SHARE=None
class Attention:
    
    
    def __init__(self,trainable=True):
        self.trainable=trainable
      
        #print ('init')


    def build(self,rgb,enc_size,read_n,T,output_shape,train_mode,conv_layer=True):

        rgb_scaled = rgb * 255.0
        batch_size=rgb.get_shape().as_list()[0]
        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [128, 128, 1]
        assert green.get_shape().as_list()[1:] == [128, 128, 1]
        assert blue.get_shape().as_list()[1:] == [128, 128, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(enc_size, state_is_tuple=True) # encoder Op
        enc_state=self.lstm_enc.zero_state(batch_size, tf.float32)
        h_enc_prev=tf.zeros((batch_size,enc_size))
        for t in range(1):
            global DO_SHARE
            r=self.read_attn(bgr,h_enc_prev,read_n,DO_SHARE)
            r=tf.image.resize_images(r,(224,224))
            vgg=vgg16.Vgg16()
            vgg.build(r,DO_SHARE)
            r=vgg.prob
            h_enc,enc_state=self.encode(enc_state,tf.concat(1,[r,h_enc_prev]),DO_SHARE)
            h_enc_prev=h_enc 
            DO_SHARE=True
        i=tf.constant(1)
        def body(i,h_enc_prev,enc_state):
            global DO_SHARE
            r=self.read_attn(bgr,h_enc_prev,read_n,DO_SHARE)
            r=tf.image.resize_images(r,(224,224))
            vgg=vgg16.Vgg16()
            vgg.build(r,DO_SHARE)
            r=vgg.prob
            h_enc,enc_state=self.encode(enc_state,tf.concat(1,[r,h_enc_prev]),DO_SHARE)
            h_enc_prev=h_enc 
          #  DO_SHARE=True # from now on, share variables
            i=tf.add(i,1)
            return i,h_enc_prev,enc_state
        def condition(i,h_enc_prev,enc_state):
            return tf.less(i,T)
        i,h_enc_prev,enc_state=tf.while_loop(condition,body,loop_vars=[i,h_enc_prev,enc_state],back_prop=True,swap_memory=False)
        ''' for t in range(T):
            r=self.read_attn(bgr,h_enc_prev,read_n,DO_SHARE)
            r=tf.image.resize_images(r,(224,224))
            vgg=vgg16.Vgg16()
            vgg.build(r,DO_SHARE)
            r=vgg.prob
            h_enc,enc_state=self.encode(enc_state,tf.concat(1,[r,h_enc_prev]),DO_SHARE)
            h_enc_prev=h_enc 
            DO_SHARE=True # from now on, share variables'''
        #print h_enc.get_shape().as_list()
        self.fc1 = self.fc_layer(enc_state[1],enc_state[1].get_shape().as_list()[1], 256, "fc1",DO_SHARE=None)
        self.relu1 = tf.nn.relu(self.fc1)
        if train_mode is not None:
            self.relu1 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu1, 0.5), lambda: self.relu1)
        elif self.trainable:
            self.relu1 = tf.nn.dropout(self.relu1, 0.5)
        self.v_i=i
        self.fc2 = self.fc_layer(self.relu1, 256, output_shape, "fc2",DO_SHARE=None)
        self.prob = tf.nn.softmax(self.fc2, name="prob")

    def linear(self,x,output_dim):
        w=tf.get_variable("w", [x.get_shape()[1], output_dim],initializer=tf.random_normal_initializer(0.0, 0.001)) 
        b=tf.get_variable("b", [output_dim], initializer=tf.random_normal_initializer(0.0, 0.001))
        return tf.matmul(x,w)+b
    def filterbank(self,gx, gy, sigma2,delta, N):

        grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
        mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
        a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
        b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
        mu_x = tf.reshape(mu_x, [-1, N, 1])
        mu_y = tf.reshape(mu_y, [-1, N, 1])
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2)) # 2*sigma2?
        Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2)) # batch x N x B
        # normalize, sum over A and B dims
        Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
        Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
        return Fx,Fy
    def encode(self,state,input,DO_SHARE):
        with tf.variable_scope("encoder",reuse=DO_SHARE):
            return self.lstm_enc(input,state)
    def attn_window(self,scope,h_dec,N,DO_SHARE):
       
        with tf.variable_scope(scope,reuse=DO_SHARE):
            params=self.linear(h_dec,5)
        gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
        gx=(A+1)/2*(gx_+1)
        gy=(B+1)/2*(gy_+1)
        sigma2=tf.exp(log_sigma2)
        delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N
        return self.filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)
    def read_attn(self,x,h_enc_prev,read_n,DO_SHARE):
     
        Fx,Fy,gamma=self.attn_window("read",h_enc_prev,read_n,DO_SHARE)
      
        def filter_img(img,Fx,Fy,gamma,N):
            Fxt=tf.transpose(Fx,perm=[0,2,1])
            img=tf.reshape(img,[-1,B,A])
           
            glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt))
            glimpse=tf.reshape(glimpse,[-1,N*N])
        
            return tf.reshape(glimpse*tf.reshape(gamma,[-1,1]),(-1,read_n,read_n,1))
        x=tf.concat(3,[filter_img(x[:,:,:,0],Fx,Fy,gamma,read_n),filter_img(x[:,:,:,1],Fx,Fy,gamma,read_n),filter_img(x[:,:,:,2],Fx,Fy,gamma,read_n)]) # batch x (read_n*read_n)
      
        return x # concat along feature axis

    def fc_layer(self, bottom, in_size, out_size, name,DO_SHARE):
        with tf.variable_scope(name,reuse=DO_SHARE):
            weights, biases = self.get_fc_var(in_size, out_size, name)
         
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        return fc

   

    def get_fc_var(self, in_size, out_size, name):
    
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")
        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        value = initial_value
        if self.trainable:
            var = tf.get_variable(var_name,initializer=value) #get_variable, change the boolian to numpy
        else:
            var = tf.get_variable(var_name,initializaer=tf.constant(value),trainable=False)
        return var
