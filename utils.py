
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy

import sys
from os import listdir
from os.path import isfile, join

from keras.utils import np_utils

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile


height, width, dim = 32, 32, 3
classes = 10

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=False,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

# this function is provided from the official site
def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

# from PIL import Image
# def ndarray2image (arr_data, image_fn):
#   img = Image.fromarray(arr_data, 'RGB')
#   img.save(image_fn)

# need pillow package
from scipy.misc import imsave
def ndarray2image (arr_data, image_fn):
    imsave(image_fn, arr_data)

def read_dataset(dataset_path, output_type,dtype=dtypes.float32,reshape=False,seed=None):
    # define the information of images which can be obtained from official website

    ''' read training data '''
    # get the file names which start with "data_batch" (training data)
    train_fns = [fn for fn in listdir(dataset_path) if isfile(join(dataset_path, fn)) & fn.startswith("data_batch")]

    # list sorting
    train_fns.sort()

    # make a glace about the training data
    fn = train_fns[0]
    raw_data = unpickle(dataset_path + fn)

    # type of raw data
    type(raw_data)
    # <type 'dict'>

    # check keys of training data
    raw_data_keys = raw_data.keys()
    # output ['data', 'labels', 'batch_label', 'filenames']

    # check dimensions of ['data']
    raw_data['data'].shape
    # (10000, 3072)

    # concatenate pixel (px) data into one ndarray [img_px_values]
    # concatenate label data into one ndarray [img_lab]
    img_px_values = 0
    img_lab = 0
    for fn in train_fns:
        raw_data = unpickle(dataset_path + fn)
        if fn == train_fns[0]:
            img_px_values = raw_data['data']
            img_lab = raw_data['labels']
        else:
            img_px_values = numpy.vstack((img_px_values, raw_data['data']))
            img_lab = numpy.hstack((img_lab, raw_data['labels']))

    X_train = []
    
    if (output_type == "vec"):
        # set X_train as 1d-ndarray (50000,3072)
        X_train = img_px_values
    elif (output_type == "img"):
        # set X_train as 3d-ndarray (50000,32,32,3)
        X_train = numpy.asarray([numpy.dstack((r[0:(width*height)].reshape(height,width),
                                               r[(width*height):(2*width*height)].reshape(height,width),
                                               r[(2*width*height):(3*width*height)].reshape(height,width)
                                             )) for r in img_px_values])
    else:
        sys.exit("Error output_type")

    Y_train = np_utils.to_categorical(numpy.array(img_lab), classes)

    # check is same or not!
    # lab_eql = numpy.array_equal([(numpy.argmax(r)) for r in Y_train], numpy.array(img_lab))

    # draw one image from the pixel data
    if (output_type == "img"):
        ndarray2image(X_train[0],"test_image.png")

    # print the dimension of training data
    print ('X_train shape:', X_train.shape)
    print ('Y_train shape:', Y_train.shape)

    ''' read testing data '''
    # get the file names which start with "test_batch" (testing data)
    test_fns = [fn for fn in listdir(dataset_path) if isfile(join(dataset_path, fn)) & fn.startswith("test_batch")]

    # read testing data
    fn = test_fns[0]
    raw_data = unpickle(dataset_path + fn)
    print ('testing file', dataset_path + fn)

    # type of raw data
    type(raw_data)

    # check keys of testing data
    raw_data_keys = raw_data.keys()
    # ['data', 'labels', 'batch_label', 'filenames']

    img_px_values = raw_data['data']

    # check dimensions of data
    print ("dim(data)", numpy.array(img_px_values).shape)
    # dim(data) (10000, 3072)

    img_lab = raw_data['labels']
    # check dimensions of labels
    print ("dim(labels)",numpy.array(img_lab).shape)
    # dim(data) (10000,)

    if (output_type == "vec"):
        X_test = img_px_values
    elif (output_type == "img"):
        X_test = numpy.asarray([numpy.dstack((r[0:(width*height)].reshape(height,width),
                                              r[(width*height):(2*width*height)].reshape(height,width),
                                              r[(2*width*height):(3*width*height)].reshape(height,width)
                                            )) for r in img_px_values])
    else:
        sys.exit("Error output_type")

    Y_test = np_utils.to_categorical(numpy.array(raw_data['labels']), classes)

    # scale image data to range [0, 1]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # print the dimension of training data
    print ('X_test shape:', X_test.shape)
    print ('Y_test shape:', Y_test.shape)

    options = dict(dtype=dtype, reshape=reshape, seed=seed)
  
    train = DataSet(X_train, Y_train, **options)
    test = DataSet(X_test, Y_test, **options)
    return train, test


import csv
def write_csv(output_fn, fit_log):
    history_fn = output_fn + '.csv'
    with open(history_fn, 'w') as csv_file:
        w = csv.writer(csv_file, lineterminator='\n')
        temp = numpy.array(list(fit_log.history.values()))
        w.writerow(list(fit_log.history.keys()))
        for i in range(temp.shape[1]):
            w.writerow(temp[:,i])
