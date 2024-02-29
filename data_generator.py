import numpy as np
import tensorflow as tf
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_paths, labels, batch_size=32, dim=(224,224), n_channels=3, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.file_paths = file_paths
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.file_paths))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

    def __data_generation(self, file_paths_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self.dim, self.n_channels))
      y = np.empty((self.batch_size,2))

      # Generate data
      for i, file_path in enumerate(file_paths_temp):
          # Store sample
          img = tf.io.read_file(file_path)
          img = tf.io.decode_png(img, channels=self.n_channels)
          img = tf.image.resize(img, [*self.dim])
          X[i,] = img.numpy()

          # Store class
          y[i] = self.labels[i]

      return X, y

    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch. Slice the list of indices Lst[Starts : End : Step]
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size: 1]

      # Find list of IDs
      list_paths_temp = [self.file_paths[k] for k in indexes]

      # Generate data
      X, y = self.__data_generation(list_paths_temp)

      return X, y