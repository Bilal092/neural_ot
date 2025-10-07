import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
from torchvision import datasets
import scipy.io as sio

import jax.numpy as jnp
import jax
from typing import Tuple
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import random


from mpl_toolkits.mplot3d import Axes3D
# parts are taken from Leititia et al. (2020) and adapted to work with JAX

def make_environment(images, labels, e, key):
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    labels = (labels > 1).astype(np.float32)  # 1 if digit > 1, else 0
    rng = np.random.RandomState(0)
    flip = (rng.rand(len(labels)) < e).astype(np.float32)
    colors = np.abs(labels - flip)  # XOR
    images = np.stack([images, images], axis=1)
    for i in range(len(images)):
        images[i, int(1 - colors[i])] = 0  # Zero out one channel
    images = images.astype(np.float32) / 255.0
    return {
        'images': jnp.array(images),
        'labels': jnp.array(labels[:, None])
    }

def make_data(dataset='mnist', key=None):
    if dataset == "mushrooms":
        x, t = load_svmlight_file("data/mushrooms.txt")
        x = x.toarray()
        x = np.delete(x, 77, 1)
        t = np.where(t == 2, 0, 1)
    elif dataset == "shuttle":
        x_train, t_train = load_svmlight_file('data/shuttle.scale')
        x_test, t_test = load_svmlight_file('data/shuttle.scale.t')
        x = np.concatenate([x_train.toarray(), x_test.toarray()])
        t = np.concatenate([t_train, t_test])
        t = (t == 1).astype(np.int32)
    elif dataset == "pageblocks":
        data = np.loadtxt('data/page-blocks.data')
        x, t = data[:, :-1], data[:, -1]
        t = (t == 1).astype(np.int32)
    elif dataset == "usps":
        x_train, t_train = load_svmlight_file('data/usps')
        x_test, t_test = load_svmlight_file('data/usps.t')
        x = np.concatenate([x_train.toarray(), x_test.toarray()])
        t = np.concatenate([t_train, t_test])
        t = np.where(t == 1, 1, 0)
    elif dataset == "connect-4":
        x, t = load_svmlight_file('data/connect-4.txt')
        x = x.toarray()
        t = (t == 1).astype(np.int32)
    elif dataset == "spambase":
        data = np.loadtxt('data/spambase.data', delimiter=',')
        x, t = data[:, :-1], data[:, -1]
    elif dataset.startswith("mnist"):
        ds = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
        images, labels = tfds.as_numpy(ds)
        if dataset == "mnist":
            env = make_environment(images[::2], labels[::2], e=0.0, key=key)
        elif dataset == "mnist_color_change_p":
            env = make_environment(images[::2], labels[::2], e=0.1, key=key)
        elif dataset == "mnist_color_change_u":
            env = make_environment(images[::2], labels[::2], e=1.0, key=key)
        data = env['images']
        x = data.reshape((data.shape[0], -1))
        t = env['labels'].flatten()
    elif dataset.startswith("surf"):
        domain = dataset[5:]
        mat = sio.loadmat(f"data/{domain}_zscore_SURF_L10.mat")
        x = mat['Xs'] if domain == "dslr" else mat['Xt']
        t = mat['Ys'] if domain == "dslr" else mat['Yt']
        t = (t == 1).astype(np.int32).flatten()
        pca = PCA(n_components=10, random_state=0)
        x = pca.fit_transform(x)
    elif dataset.startswith("decaf"):
        domain = dataset[6:]
        mat = sio.loadmat(f"data/{domain}_decaf.mat")
        x = mat['feas']
        t = (mat['labels'] == 1).astype(np.int32).flatten()
        pca = PCA(n_components=40, random_state=0)
        x = pca.fit_transform(x)
    else:
        raise ValueError("Unknown dataset")

    return jnp.array(x), jnp.array(t)
    
    
def normalize_minmax(x):
    min_x = jnp.min(x, axis=0)
    max_x = jnp.max(x, axis=0)
    div = jnp.where((max_x - min_x) == 0, 1.0, max_x - min_x)
    return (x - min_x) / div


def sample_indices(key, x, y, num_samples):
    idx = jax.random.permutation(key, len(x))
    return x[idx[:num_samples]], y[idx[:num_samples]], x[idx[num_samples:]], y[idx[num_samples:]]
    
    
def create_dataset_tf(dataset, train_ratio=0.9, batch_size=32, additional_dim=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    num_devices = jax.local_device_count()
    per_device_batch_size = batch_size // num_devices
    shuffle_buffer_size = 100000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = 1
    
    # Create additional data dimension when jitting multiple steps together 
    # we are jitting one step but with larger memory one can jit multiple batches together. 
    
    if additional_dim is None:
        batch_dims = [jax.local_device_count(), per_device_batch_size]
    else:
        batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size]

    data, labels = make_data(dataset, key=random.key(0))
    data = normalize_minmax(data)

    # Shuffle the data
    indices = np.random.permutation(len(data))
    data, labels = data[indices], labels[indices]
    sz = data.shape[1]

    # Train-validation split
    n_train = int(train_ratio * len(data))
    data_train, data_val = data[:n_train], data[n_train:]
    labels_train, labels_val = labels[:n_train], labels[n_train:]

    # Split train into two equal parts
    n_half = n_train // 2
    data_pos_part, data_unlabeled_part = data_train[:n_half], data_train[n_half:]
    labels_pos_part, labels_unlabeled_part = labels_train[:n_half], labels_train[n_half:]

    # Select only positive samples from the first half
    mask_positive = labels_pos_part == 1
    data_pos = data_pos_part[mask_positive]
    labels_pos = labels_pos_part[mask_positive]

    # Helper to create batched, repeated, prefetched dataset
    # def make_dataset(data, labels, shuffle=True):
    #     ds = tf.data.Dataset.from_tensor_slices((data, labels))
    #     if shuffle:
    #         ds = ds.shuffle(buffer_size=len(data))
    #     return ds.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def make_dataset(arr, shuffle=False):
            dataset = tf.data.Dataset.from_tensor_slices(arr)
            dataset_options = tf.data.Options()
            dataset_options.experimental_optimization.map_parallelization = True
            dataset_options.threading.private_threadpool_size = 48
            dataset_options.threading.max_intra_op_parallelism = 1
            # dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.with_options(dataset_options)
            dataset = dataset.repeat(count=num_epochs)
            if shuffle:
                dataset = dataset.shuffle(shuffle_buffer_size)
            for batch_size in reversed(batch_dims):
                dataset = dataset.batch(batch_size, drop_remainder=True)
            return dataset.prefetch(prefetch_size)

    # Construct datasets
    ds_p = make_dataset((data_pos, labels_pos), shuffle=True)
    ds_ul = make_dataset((data_unlabeled_part, labels_unlabeled_part), shuffle=True)
    ds_val = make_dataset((data_val, labels_val), shuffle=False)  # No repeat for val

    return ds_p, ds_ul, ds_val, sz
