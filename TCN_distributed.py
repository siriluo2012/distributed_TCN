import numpy as np
import glob
import pandas as pd
import argparse

import time
from sklearn.utils import shuffle

import tensorflow as tf
import keras.backend as K
from tensorflow.keras.layers import Dense, TimeDistributed
import tensorflow.keras.optimizers as keras_optimizers
from tensorflow.keras import Input, Model

from tcn import TCN
from tcn import compiled_tcn

def loadData(path, batch_size=32):
    x_append = []
    y_append = []

    SF_f = 1 # Scale flux
    SF_d = 100 # Scale disp
    SF_T = 1/1000 # Scale Temp
    SF_S = 1/1e7 # Scale Stress

    path = path
    all_files = glob.glob(path + "Data/*.txt")
    shuffle(all_files)

    for filename in all_files:
        file_read = pd.read_csv(filename, sep="\s+", header=None)
        file_read = file_read.values
        file_read = np.array(file_read)
        x_f = SF_f * np.asarray(file_read[:,:1])
        x_d = SF_d * np.asarray(file_read[:,1:2])
        x_ = np.concatenate((x_f, x_d), axis=1)
        y_T = SF_T * file_read[:,2:6]
        y_S = SF_S * file_read[:,6:]
        y_ = np.concatenate((y_T, y_S), axis=1)
        ymax = y_.max()
        ymin = y_.min()
        if np.isnan(ymax)==False and np.isnan(ymin)==False:
            x_append.append(x_)
            y_append.append(y_)

    x_data = np.asarray(x_append)
    y_data = np.asarray(y_append)


    batch_size = batch_size
    x = x_data
    y = y_data
    num_examples = x.shape[0]

    train_fraction = 0.8

    train_range    = int(num_examples * train_fraction)
    x_train        = x[:train_range,:,:]
    y_train        = y[:train_range,:,:]

    mod = x_train.shape[0] % (batch_size * ngpus)
    x_train = x_train[:-mod, :, :]
    y_train = y_train[0:-mod, :, :]

    x_val        = x[train_range:,:,:]
    y_val        = y[train_range:,:,:]

    mod = x_val.shape[0] % (batch_size * ngpus)
    x_val = x_val[:-mod, :, :]
    y_val = y_val[:-mod, :, :]

    return x_train, y_train, x_val, y_val

def get_compiled_model(batch_size=32, timesteps = 101, input_dim = 2):

    batch_size, timesteps, input_dim = batch_size, timesteps, input_dim

    i = Input(batch_shape=(batch_size, timesteps, input_dim))

    o = TCN(nb_filters=64, nb_stacks=1, dilations=(1, 2, 4, 8, 16),
          dropout_rate=0.0, kernel_size=7, kernel_initializer='glorot_uniform',
          use_batch_norm=False, use_layer_norm=False, padding='causal', activation='relu',
          return_sequences=True, use_skip_connections=True)(i)

    o = TCN(nb_filters=128, nb_stacks=1, dilations=(1, 2, 4, 8, 16, 32),
          dropout_rate=0.0, kernel_size=4, kernel_initializer='glorot_uniform',
          use_batch_norm=False, use_layer_norm=False, padding='causal', activation='relu',
          return_sequences=True, use_skip_connections=True)(o)

    o = TCN(nb_filters=256, nb_stacks=1, dilations=(1, 2, 4, 8, 16, 32, 64),
          dropout_rate=0.0, kernel_size=2, kernel_initializer='glorot_uniform',
          use_batch_norm=False, use_layer_norm=False, padding='causal', activation='relu',
          return_sequences=True, use_skip_connections=True)(o)  # The TCN layers are here.

    o = TimeDistributed(Dense(8))(o)
    m = Model(inputs=[i], outputs=[o])

    optimizer = keras_optimizers.Adam(learning_rate=0.001, clipnorm=1.)

    m.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse'])

    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataPath', type=str, default='/home/shirui/student_consulting/seid_distributed/')
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    path = args.dataPath


    # set up the distributed training

    gpus = tf.config.list_physical_devices('GPU')
    ngpus = len(gpus)
    devices_names = [d.name.split('e:')[1] for d in gpus]

    strategy = tf.distribute.MirroredStrategy(devices=devices_names)
    with strategy.scope():
        model = get_compiled_model(batch_size=batch_size)

    # create batch dataset
    x_train, y_train, x_val, y_val = loadData(path, batch_size)

    BATCH_SIZE_PER_REPLICA = batch_size
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    BUFFER_SIZE = 10000

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,  y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    #train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    #val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # start training
    t0 = time.time()

    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=2)
    total_time = time.time() - t0

    print('total time spent on training: ', total_time)
