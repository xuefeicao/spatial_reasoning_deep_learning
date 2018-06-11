import tensorflow as tf
import numpy as np
from scipy import misc
from glob import glob

def get_image_size(config):
    im_size = misc.imread(glob(config.train_directory + '*' + config.im_ext)[0]).shape
    if len(im_size) == 2:
        im_size = np.hstack((im_size,3))
    return im_size

def read_and_decode_single_example(filename):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([784], tf.int64)
        })
    # now return the converted data
    label = features['label']
    image = features['image']
    return label, image

def read_and_decode(filename_queue,im_size,model_input_shape,train):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  #flat_shape = im_size[0] * im_size[1] * im_size[2]
  features = tf.parse_single_example(
    serialized_example,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string) #flat_shape * 4 (32-bit flaot -> bytes) = 1080000
    })

  # Convert from a scalar string tensor (whose single string has
  image = tf.decode_raw(features['image'], tf.float32)
  image = tf.transpose(tf.reshape(image, np.array(im_size)[[2,0,1]]), [2,1,0]) #Need to reconstruct channels first then transpose channels
  image.set_shape(im_size)

  # Insert augmentation and preprocessing here
  if train is not None:
    if 'left_right' in train:
      image = tf.image.random_flip_left_right(image)
    if 'up_down' in train:
      image = tf.image.random_flip_up_down(image)
    if 'random_crop' in train:
      hw = np.random.randint(np.round(im_size[0]*.8),im_size[0],1)[0]#tf.cast(tf.round(tf.mul(tf.constant(im_size[0],dtype=tf.float32),tf.random_uniform(1,minval=.8,maxval=1))),tf.int32)
      image = tf.random_crop(image, [hw,hw,im_size[2]])
    if 'random_brightness' in train:
      image = tf.image.random_brightness(image, max_delta=32./255.)
    if 'random_contrast' in train:
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    if 'rotate' in train:
      image = tf.image.rot90(image, k=np.random.randint(4))

  image = tf.image.resize_images(image,(model_input_shape[0],model_input_shape[1]))

  # If necessary Convert from [0, 255] -> [0, 1] floats. x * (1. / 255)
  image = tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 1.0) #Make sure to clip the 

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label

def inputs(tfrecord_file, batch_size, im_size, model_input_shape, train=None, num_epochs=None):
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [tfrecord_file], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue,im_size,model_input_shape,train)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels



        
