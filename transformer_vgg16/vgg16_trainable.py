import os,sys
#sys.path.append('/transformer')
import tensorflow as tf
#from transformer.spatial_transformer import transformer
import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    """
    A trainable version VGG16.
    """

    def __init__(self, vgg16_npy_path=None, trainable=True, fine_tune_layers=None):
        if vgg16_npy_path is not None:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            if fine_tune_layers is not None: #pop the specified keys from the weights that will be loaded
                for key in fine_tune_layers:
                    del self.data_dict[key]
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    def build(self, rgb, spatial_size,output_shape = None, train_mode=None,spatial_layer=True):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        if output_shape is None:
            output_shape = 1000

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        if spatial_layer== True:
            self.theta_0=tf.nn.tanh(self.fc_layer(self.conv5_3,100352,6 ,'localisation_0'))
            if train_mode is not None: #Consider changing these to numpy conditionals
                self.theta_0 = tf.cond(train_mode, lambda: tf.nn.dropout(self.theta_0, 0.5), lambda: self.theta_0)
	    self.theta=self.fc_layer(self.theta_0,6,6 ,'localisation_1',spatial=spatial_layer)
            if train_mode is not None: #Consider changing these to numpy conditionals
                self.theta = tf.cond(train_mode, lambda: tf.nn.dropout(self.theta, 0.5), lambda: self.theta)
            #self.theta=self.fc_layer(self.conv5_3,100352,6 ,'localisation_1')
            
	    self.trans = self.transformer(self.conv5_3, self.theta, spatial_size)
     
            self.pool5 = self.max_pool(self.trans, 'pool5')
            self.fc6 = self.fc_layer(self.pool5, spatial_size[1]/2*spatial_size[0]/2*512, 4096, "fc6") 
        else:
            self.pool5 = self.max_pool(self.conv5_3, 'pool5')
            self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6") 

        
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None: #Consider changing these to numpy conditionals
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, 0.5), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, 0.5), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)

        self.fc8 = self.fc_layer(self.relu7, 4096, output_shape, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name,spatial=False):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name,spatial)
            x = tf.reshape(bottom, [-1, in_size])
            if spatial==False:
                fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            else:
                
                fc = tf.nn.tanh(tf.matmul(x, weights) + biases)
            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name,spatial=False):
        
        if spatial==False:
            initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
            weights = self.get_var(initial_value, name, 0, name + "_weights")
            initial_value = tf.truncated_normal([out_size], .0, .001)
        else:
            initial_value = tf.zeros([in_size, out_size], dtype='float32')
            weights = self.get_var(initial_value, name, 0, name + "_weights")
            initial = np.array([[1., 0, 0], [0, 1., 0]])
            initial = initial.astype('float32')
            initial_value = initial.flatten()
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name) #get_variable, change the boolian to numpy
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        #assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg16-save.npy"):
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
    def transformer(self,U,theta, out_size, name='SpatialTransformer', **kwargs):
	    def _repeat(x, n_repeats):
		with tf.variable_scope('_repeat'):
		    rep = tf.transpose(
		        tf.expand_dims(tf.ones(shape=tf.pack([n_repeats, ])), 1), [1, 0])
		    rep = tf.cast(rep, 'int32')
		    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
		    return tf.reshape(x, [-1])

	    def _interpolate(im, x, y, out_size):
		with tf.variable_scope('_interpolate'):
		    # constants
		    num_batch = tf.shape(im)[0]
		    height = tf.shape(im)[1]
		    width = tf.shape(im)[2]
		    channels = tf.shape(im)[3]
                    
		    x = tf.cast(x, 'float32')
		    y = tf.cast(y, 'float32')
		    height_f = tf.cast(height, 'float32')
		    width_f = tf.cast(width, 'float32')
		    out_height = out_size[0]
		    out_width = out_size[1]
		    zero = tf.zeros([], dtype='int32')
		    max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
		    max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

		    # scale indices from [-1, 1] to [0, width/height]
		    x = (x + 1.0)*(width_f) / 2.0
		    y = (y + 1.0)*(height_f) / 2.0

		    # do sampling
		    x0 = tf.cast(tf.floor(x), 'int32')
		    x1 = x0 + 1
		    y0 = tf.cast(tf.floor(y), 'int32')
		    y1 = y0 + 1

		    x0 = tf.clip_by_value(x0, zero, max_x)
		    x1 = tf.clip_by_value(x1, zero, max_x)
		    y0 = tf.clip_by_value(y0, zero, max_y)
		    y1 = tf.clip_by_value(y1, zero, max_y)
		    dim2 = width
		    dim1 = width*height
		    base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
		    base_y0 = base + y0*dim2
		    base_y1 = base + y1*dim2
		    idx_a = base_y0 + x0
		    idx_b = base_y1 + x0
		    idx_c = base_y0 + x1
		    idx_d = base_y1 + x1

		    # use indices to lookup pixels in the flat image and restore
		    # channels dim
		    im_flat = tf.reshape(im, tf.pack([-1, channels]))
		    im_flat = tf.cast(im_flat, 'float32')
		    Ia = tf.gather(im_flat, idx_a)
		    Ib = tf.gather(im_flat, idx_b)
		    Ic = tf.gather(im_flat, idx_c)
		    Id = tf.gather(im_flat, idx_d)

		    # and finally calculate interpolated values
		    x0_f = tf.cast(x0, 'float32')
		    x1_f = tf.cast(x1, 'float32')
		    y0_f = tf.cast(y0, 'float32')
		    y1_f = tf.cast(y1, 'float32')
		    wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
		    wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
		    wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
		    wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
		    output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
		    return output

	    def _meshgrid(height, width):
		with tf.variable_scope('_meshgrid'):
		    # This should be equivalent to:
		    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
		    #                         np.linspace(-1, 1, height))
		    #  ones = np.ones(np.prod(x_t.shape))
		    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
		    x_t = tf.matmul(tf.ones(shape=tf.pack([height, 1])),
		                    tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
		    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
		                    tf.ones(shape=tf.pack([1, width])))

		    x_t_flat = tf.reshape(x_t, (1, -1))
		    y_t_flat = tf.reshape(y_t, (1, -1))

		    ones = tf.ones_like(x_t_flat)
		    grid = tf.concat(0, [x_t_flat, y_t_flat, ones])
		    return grid

	    def _transform(theta, input_dim, out_size):
		with tf.variable_scope('_transform'):
		    num_batch = tf.shape(input_dim)[0]
		    height = tf.shape(input_dim)[1]
		    width = tf.shape(input_dim)[2]
		    num_channels = tf.shape(input_dim)[3]
		    theta = tf.reshape(theta, (-1, 2, 3))
		    theta = tf.cast(theta, 'float32')

		    # grid of (x_t, y_t, 1), eq (1) in ref [1]
		    height_f = tf.cast(height, 'float32')
		    width_f = tf.cast(width, 'float32')
		    out_height = out_size[0]
		    out_width = out_size[1]
		    grid = _meshgrid(out_height, out_width)
		    grid = tf.expand_dims(grid, 0)
		    grid = tf.reshape(grid, [-1])
		    grid = tf.tile(grid, tf.pack([num_batch]))
		    grid = tf.reshape(grid, tf.pack([num_batch, 3, -1]))

		    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
		    T_g = tf.batch_matmul(theta, grid)
		    x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
		    y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
		    x_s_flat = tf.reshape(x_s, [-1])
		    y_s_flat = tf.reshape(y_s, [-1])

		    input_transformed = _interpolate(
		        input_dim, x_s_flat, y_s_flat,
		        out_size)

		    output = tf.reshape(
		        input_transformed, tf.pack([num_batch, out_height, out_width, num_channels]))
        
		    return output

	    with tf.variable_scope(name):
		output = _transform(theta, U, out_size)
                
		return output

