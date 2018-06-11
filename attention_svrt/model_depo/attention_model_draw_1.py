import os,sys
import tensorflow as tf
import numpy as np
import time
import inspect
A=128
B=128
eps=1e-8

VGG_MEAN = [103.939, 116.779, 123.68]
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
        rgb=tf.div(tf.add(tf.add(bgr[:,:,:,0],bgr[:,:,:,1]),bgr[:,:,:,2]),3.0)
        #print rgb.get_shape()
        output = tf.Variable(tf.zeros([batch_size, enc_size]), trainable=False)
        state = tf.Variable(tf.zeros([batch_size, enc_size]), trainable=False)
    
        DO_SHARE=None
        for t in range(T):
            r=self.read_attn(rgb,output,read_n,DO_SHARE)
            if conv_layer==True:
                r=tf.reshape(r,[-1,read_n,read_n,1])
                r=tf.image.resize_images(r,(28,28))
                r=self.c_l(r,read_n,DO_SHARE)
     
            output, state =self.encode(r,output,state,DO_SHARE,read_n*read_n,enc_size)
            DO_SHARE=True # from now on, share variables
        #print h_enc.get_shape().as_list()
        self.fc1 = self.fc_layer(state,state.get_shape().as_list()[1], 256, "fc1",DO_SHARE=None)
        self.relu1 = tf.nn.relu(self.fc1)
        if train_mode is not None:
            self.relu1 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu1, 0.5), lambda: self.relu1)
        elif self.trainable:
            self.relu1 = tf.nn.dropout(self.relu1, 0.5)

        self.fc2 = self.fc_layer(self.relu1, 256, output_shape, "fc2",DO_SHARE=None)
        self.prob = tf.nn.softmax(self.fc2, name="prob")
        #print [a.name for a in tf.get_collection(tf.GraphKeys.VARIABLES, scope='encoder')]
    def c_l(self,x,N,DO_SHARE):
        self.conv1_1 = self.conv_layer(x, 1, 6, "conv1_1",DO_SHARE)
        #self.conv1_2 = self.conv_layer(self.conv1_1, 6, 6, "conv1_2",DO_SHARE)
        self.pool1 = self.max_pool(self.conv1_1, 'pool1')
      
        self.conv2_1 = self.conv_layer(self.pool1, 6, 16, "conv2_1",DO_SHARE)
        #self.conv2_2 = self.conv_layer(self.conv2_1, 12, 12, "conv2_2",DO_SHARE)
        self.pool2 = self.max_pool(self.conv2_1, 'pool2')

        self.fc11=self.fc_layer(self.pool2,28*28/16*16,N*N,'fc11',DO_SHARE)
        #self.fc12=self.fc_layer(self.fc11,120,N*N,'fc12',DO_SHARE)
        return self.fc11
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name,DO_SHARE):
        with tf.variable_scope(name,reuse=DO_SHARE):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu
    def linear(self,x,output_dim):
        w=tf.get_variable("w", [x.get_shape()[1], output_dim]) 
        b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
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
    def encode(self,input,output,state,DO_SHARE,vocabulary_size,num_nodes):
        with tf.variable_scope("encoder",reuse=DO_SHARE):
        	  # Input gate: input, previous output, and bias.
		ix = tf.get_variable('ix',initializer=tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
		im = tf.get_variable('im',initializer=tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
		ib = tf.get_variable('ib',initializer=tf.zeros([1, num_nodes]))
		  # Forget gate: input, previous output, and bias.
		fx = tf.get_variable('fx',initializer=tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
		fm = tf.get_variable('fm',initializer=tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
		fb = tf.get_variable('fb',initializer=tf.zeros([1, num_nodes]))
		  # Memory cell: input, state and bias.                             
		cx = tf.get_variable('cx',initializer=tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
		cm = tf.get_variable('cm',initializer=tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
		cb = tf.get_variable('cb',initializer=tf.zeros([1, num_nodes]))
		# Output gate: input, previous output, and bias.
		ox = tf.get_variable('ox',initializer=tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
		om = tf.get_variable('om',initializer=tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
		ob = tf.get_variable('ob',initializer=tf.zeros([1, num_nodes]))
			  # Variables saving state across unrollings.
	def lstm_cell(i, o, state):
	    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
	    Note that in this formulation, we omit the various connections between the
	    previous state and the gates."""
	    input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
	    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
	    update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
	    state = forget_gate * state + input_gate * tf.tanh(update)
	    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
	    return output_gate * tf.tanh(state), state
	return lstm_cell(input,output,state) 
			
            
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
        #print x.get_shape()
        Fx,Fy,gamma=self.attn_window("read",h_enc_prev,read_n,DO_SHARE)
        #print Fx.get_shape()
        #print Fy.get_shape()
        #print gamma.get_shape()
        def filter_img(img,Fx,Fy,gamma,N):
            Fxt=tf.transpose(Fx,perm=[0,2,1])
            img=tf.reshape(img,[-1,B,A])
            #print tf.batch_matmul(img,Fxt).get_shape()
            glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt))
            glimpse=tf.reshape(glimpse,[-1,N*N])
           # print glimpse.get_shape()
           # print gamma.get_shape()
            return glimpse*tf.reshape(gamma,[-1,1])
        x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
        #print x.get_shape
        return x # concat along feature axis

    def fc_layer(self, bottom, in_size, out_size, name,DO_SHARE):
        with tf.variable_scope(name,reuse=DO_SHARE):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            #print weights.name
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
        value = initial_value
        if self.trainable:
            var = tf.get_variable(var_name,initializer=value) #get_variable, change the boolian to numpy
        else:
            var = tf.get_variable(var_name,initializaer=tf.constant(value, dtype=tf.float32))
        return var
