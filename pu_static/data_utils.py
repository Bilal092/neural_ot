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
import pandas as pd
import tensorflow_datasets as tfds
from jax import random
from sklearn.linear_model import LogisticRegression


# parts are taken from Chapel et al. (2020) and adapted to work with JAX https://proceedings.neurips.cc/paper_files/paper/2020/file/1e6e25d952a0d639b676ee20d0519ee2-Paper.pdf

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
    
    
def make_binary_class(y):
    if np.unique(y).shape[0]>2:
        values, counts = np.unique(y, return_counts=True)
        ind = np.argmax(counts)
        major_class = values[ind]
        for i in np.arange(y.shape[0]):
            if y[i]==major_class:
                y[i]=1
            else:
                y[i]=0
    return y

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def make_data(dataset='mnist', key=None):
    if dataset == "mushrooms":
        x, t = load_svmlight_file("/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/PU-learning/data/mushrooms.txt")
        x = x.toarray()
        x = np.delete(x, 77, 1)
        t = np.where(t == 2, 0, 1)
    elif dataset == "shuttle":
        x_train, t_train = load_svmlight_file('/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/PU-learning/data/shuttle.scale')
        x_test, t_test = load_svmlight_file('/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/PU-learning/data/shuttle.scale.t')
        x = np.concatenate([x_train.toarray(), x_test.toarray()])
        t = np.concatenate([t_train, t_test])
        t = (t == 1).astype(np.int32)
    elif dataset == "pageblocks":
        data = np.loadtxt('/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/PU-learning/data/page-blocks.data')
        x, t = data[:, :-1], data[:, -1]
        t = (t == 1).astype(np.int32)
    elif dataset == "usps":
        x_train, t_train = load_svmlight_file('/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/PU-learning/data/usps')
        x_test, t_test = load_svmlight_file('/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/PU-learning/data/usps.t')
        x = np.concatenate([x_train.toarray(), x_test.toarray()])
        t = np.concatenate([t_train, t_test])
        t = np.where(t == 1, 1, 0)
    elif dataset == "connect-4":
        x, t = load_svmlight_file('/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/PU-learning/data/connect-4.txt')
        x = x.toarray()
        #t = (t == 1).astype(np.int32)
        t = np.where(t==-1, 0, 1)
    elif dataset == "spambase":
        data = np.loadtxt('/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/PU-learning/data/spambase.data', delimiter=',')
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
        mat = sio.loadmat(f"/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/PU-learning/data/{domain}_zscore_SURF_L10.mat")
        x = mat['Xs'] if domain == "dslr" else mat['Xt']
        t = mat['Ys'] if domain == "dslr" else mat['Yt']
        t = (t == 1).astype(np.int32).flatten()
        pca = PCA(n_components=10, random_state=0)
        x = pca.fit_transform(x)
    elif dataset.startswith("decaf"):
        domain = dataset[6:]
        mat = sio.loadmat(f"/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/PU-learning/data/{domain}_decaf.mat")
        x = mat['feas']
        t = (mat['labels'] == 1).astype(np.int32).flatten()
        pca = PCA(n_components=40, random_state=0)
        x = pca.fit_transform(x)
        
    elif dataset in  ["Abalone", "Banknote", "Breast-w", "Diabetes", "Haberman", "Heart",
                                 "Ionosphere", "Isolet", "Jm1", "Kc1", "Madelon", "Musk", "Segment",
                                 "Semeion", "Sonar", "Spambase_n", "Vehicle", "Waveform", "Wdbc", "Yeast"]:
        
        df = pd.read_csv("/lustre/cniel/neural-ot-ss/Images_domain_translation/PU-Learning/data/datasets/"+dataset+".csv", sep=",")
        del df['BinClass']
        df = df.to_numpy()
        p = df.shape[1]-1
        Xall = df[:,0:p]
        yall = df[:,p]
        yall = make_binary_class(yall)
        
        # X, Xtest, y, ytest = train_test_split(Xall, yall, test_size=0.25, random_state=sym)
        # jnp.array(x), jnp.array()
        x = Xall
        t = yall

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
    # we are jitting one step but with larger and better memory througput one can jit multiple batches together. 
    
    if additional_dim is None:
        batch_dims = [jax.local_device_count(), per_device_batch_size]
    else:
        batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size]
    
    jax.debug.print("dataset {bar}", bar=dataset)
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


def create_pu_dataset_tf(config, dataset_p, dataset_u, size_p, size_u_eval, prior, seed_nb=None, additional_dim=None, seed=None):
    """
    Draw a PU dataset (SAR) and return tf.data.Dataset objects (JAX-compatible).
    batch_sizes should be devisible by number of devices to preserve all data points in training 

    Returns:
        tf.data.Dataset: positive samples (xp_ds)
        tf.data.Dataset: unlabeled samples (xu_ds)
        tf.data.Dataset: unlabeled labels (yu_ds)
        tf.data.Dataset: unlabeled zipped (xu, yu) [optional usage]
    """
    if seed is not None:
        np.random.seed(seed)
        
    num_devices = jax.local_device_count()
    # per_device_batch_size_p = size_p // num_devices
    # per_device_batch_size_u = size_u_eval // num_devices
    per_device_batch_size_p = config.train.batch_size // num_devices
    per_device_batch_size_u = config.train.batch_size // num_devices
    shuffle_buffer_size = 100000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = 1
    
    # Create additional data dimension when jitting multiple steps together 
    # we are jitting one step but with larger and better memory througput one can jit multiple batches together. 
    
    if additional_dim is None:
        batch_dims_p = [jax.local_device_count(), per_device_batch_size_p]
        batch_dims_u = [jax.local_device_count(), per_device_batch_size_u]
    else:
        batch_dims_p = [jax.local_device_count(), additional_dim, per_device_batch_size_p]
        batch_dims_u = [jax.local_device_count(), additional_dim, per_device_batch_size_u]
    
    # Load and normalize dataset
    
    if config.train.mode in ["ul_batch", "ul_dataset"]:
        x, t = make_data(dataset=dataset_p)
        x, t = np.array(x), np.array(t)
        div = np.max(x, axis=0) - np.min(x, axis=0)
        div[div == 0] = 1
        x = (x - np.min(x, axis=0)) / div
        sz = x.shape[1]
        
        xp_t = x[t == 1]
        tp_t = t[t == 1]
        xp, xp_other, tp, tp_o = train_test_split(xp_t, tp_t, train_size=size_p, random_state=seed_nb)
        yp = np.array(tp).astype(np.uint8)
    
    if config.train.mode == "ul_batch":
        size_u_p = int(prior * size_u_eval)
        size_u_n = size_u_eval - size_u_p

        if dataset_u == dataset_p:
            xup_eval, xup_train, _, yup_train = train_test_split(xp_other, tp_o, train_size=size_u_p, random_state=seed_nb)
        else:
            x, t = make_data(dataset=dataset_u)
            x, t = np.array(x), np.array(t)
            div = np.max(x, axis=0) - np.min(x, axis=0)
            div[div == 0] = 1
            x = (x - np.min(x, axis=0)) / div
            xp_other = x[t == 1]
            tp_o = t[t == 1]
            xup_eval, xup_train, _, yup_train = train_test_split(xp_other, tp_o, train_size=size_u_p, random_state=seed_nb)

        xn_t = x[t == 0]
        tn_t = t[t == 0]
        xun_eval, xun_train, _, yun_train = train_test_split(xn_t, tn_t, train_size=size_u_n, random_state=seed_nb)

        xu_eval = np.concatenate([xup_eval, xun_eval], axis=0).astype(np.float32)
        yu_eval = np.concatenate([np.ones(len(xup_eval)), np.zeros(len(xun_eval))]).astype(np.uint8)
        
        xu_train = np.concatenate([xup_train, xun_train], axis=0).astype(np.float32)
        yu_train = np.concatenate([np.ones(len(xup_train)), np.zeros(len(xun_train))]).astype(np.uint8)
        
    elif config.train.mode == "ul_dataset":
        train_ratio = config.data.train_ratio
        eval_ratio = config.data.eval_ratio
        assert abs(train_ratio + eval_ratio - 1.0) < 1e-6, "Train + Eval ratios must sum to 1"

        x_rest_pos = xp_other
        y_rest_pos = tp_o
        x_rest_neg = x[t == 0]
        y_rest_neg = t[t == 0]

        max_total_ul = min(len(x_rest_pos) + len(x_rest_neg), size_p + size_u_eval)

        total_ul_pos = int(round(prior * max_total_ul))
        total_ul_neg = max_total_ul - total_ul_pos

        x_ul_pos, _, y_ul_pos, _ = train_test_split(x_rest_pos, y_rest_pos, train_size=total_ul_pos, random_state=seed_nb)
        x_ul_neg, _, y_ul_neg, _ = train_test_split(x_rest_neg, y_rest_neg, train_size=total_ul_neg, random_state=seed_nb)

        x_ul_full = np.concatenate([x_ul_pos, x_ul_neg], axis=0)
        y_ul_full = np.concatenate([y_ul_pos, y_ul_neg], axis=0)
        idx = np.random.permutation(len(x_ul_full))
        x_ul_full, y_ul_full = x_ul_full[idx], y_ul_full[idx]

        n_train = int(train_ratio * len(x_ul_full))
        n_eval = len(x_ul_full) - n_train

        xu_train, yu_train = x_ul_full[:n_train], y_ul_full[:n_train]
        xu_eval, yu_eval = x_ul_full[n_train:], y_ul_full[n_train:]
        
    elif config.train.mode =='propensity':
        Xall, yall  = make_data(dataset=dataset_p)
        Xall, yall = np.array(Xall), np.array(yall)
        
        X, Xtest, y, ytest = train_test_split(Xall, yall, test_size=config.data.test_size, random_state=seed_nb)
        n = X.shape[0]
        sz = X.shape[1]

        # Make PU data set
        prob_true = LogisticRegression(random_state=seed_nb).fit(X, y).predict_proba(X)[:, 1]
        prob_true[np.where(prob_true==1)] = 0.999
        prob_true[np.where(prob_true==0)] = 0.001
        s = np.zeros(n)
        if config.train.label_strat == 'S1':
            prop_score = np.full(n, 0.1)
        elif config.train.label_strat == 'S2':
            prop_score = 0.1 * prob_true
        elif config.train.label_strat == 'S3':
            prop_score = sigmoid(-0.5 * prob_true - 1.5)
        elif config.train.label_strat == 'S4':
            lin_pred = np.log(prob_true/(1 - prob_true))
            prop_score = 0.5 * sigmoid(-0.5 * lin_pred)
            
        s = (y == 1) * np.random.binomial(1, prop_score)
        xp = X[s==1]
        xp = xp.astype(np.float32)
        yp = np.ones(len(xp)).astype(np.uint8)
        xu_train = X[s==0].astype(np.float32)
        yu_train = y[s==0].astype(np.uint8)
        
    def make_dataset_u(arr, shuffle=False):
        is_list_or_tuple = isinstance(arr, (list, tuple))
        is_dict = isinstance(arr, dict)

        if is_list_or_tuple or is_dict:
            if not arr:
                raise ValueError("Input container cannot be empty.")
            
            values = list(arr.values()) if is_dict else arr
            if not values: 
                raise ValueError("Input container has no arrays.")
                
            first_len = values[0].shape[0]
            if not all(sub_arr.shape[0] == first_len for sub_arr in values):
                raise ValueError("All arrays in the container must have the same length.")
            
            source_arr = values[0]
        else:
            source_arr = arr 

        if not hasattr(source_arr, 'shape'):
            raise TypeError("Input must be an array or a container of arrays.")

        current_size = source_arr.shape[0]
        if current_size == 0:
            raise ValueError("Input array(s) cannot be empty.")

        # --- Padding Logic ---
        total_batch_size = np.prod(batch_dims_u) if batch_dims_u else 1
        final_arr = arr
        remainder = current_size % total_batch_size

        if current_size < total_batch_size:
            num_to_add = total_batch_size - current_size
            temp_ds = tf.data.Dataset.from_tensor_slices(arr).repeat()
            padding = next(iter(temp_ds.batch(num_to_add)))
            if is_list_or_tuple:
                final_arr = type(arr)(tf.concat([a, p], 0) for a, p in zip(arr, padding))
            elif is_dict:
                final_arr = {k: tf.concat([arr[k], padding[k]], 0) for k in arr}
            else:
                final_arr = tf.concat([arr, padding], 0)
                
        elif remainder != 0:
            pad_needed = total_batch_size - remainder
            indices = tf.random.uniform(shape=[pad_needed], maxval=current_size, dtype=tf.int32)
            if is_list_or_tuple:
                padding = type(arr)(tf.gather(a, indices) for a in arr)
                final_arr = type(arr)(tf.concat([a, p], 0) for a, p in zip(arr, padding))
            elif is_dict:
                padding = {k: tf.gather(v, indices) for k, v in arr.items()}
                final_arr = {k: tf.concat([arr[k], padding[k]], 0) for k in arr}
            else:
                padding = tf.gather(arr, indices)
                final_arr = tf.concat([arr, padding], 0)

        # --- Dataset Creation ---
        dataset = tf.data.Dataset.from_tensor_slices(final_arr)
        if num_epochs != 1:
            dataset = dataset.repeat(count=num_epochs)
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        
        for batch_size in reversed(batch_dims_p):
            dataset = dataset.batch(batch_size, drop_remainder=False)
            
        return dataset.prefetch(tf.data.AUTOTUNE)
    
        
    def make_dataset_p(arr, shuffle=False):
        is_list_or_tuple = isinstance(arr, (list, tuple))
        is_dict = isinstance(arr, dict)

        if is_list_or_tuple or is_dict:
            if not arr:
                raise ValueError("Input container cannot be empty.")
            
            values = list(arr.values()) if is_dict else arr
            if not values: 
                raise ValueError("Input container has no arrays.")
                
            first_len = values[0].shape[0]
            if not all(sub_arr.shape[0] == first_len for sub_arr in values):
                raise ValueError("All arrays in the container must have the same length.")
            
            source_arr = values[0]
        else:
            source_arr = arr 

        if not hasattr(source_arr, 'shape'):
            raise TypeError("Input must be an array or a container of arrays.")

        current_size = source_arr.shape[0]
        if current_size == 0:
            raise ValueError("Input array(s) cannot be empty.")

        # --- Padding Logic ---
        total_batch_size = np.prod(batch_dims_p) if batch_dims_p else 1
        final_arr = arr
        remainder = current_size % total_batch_size

        if current_size < total_batch_size:
            num_to_add = total_batch_size - current_size
            temp_ds = tf.data.Dataset.from_tensor_slices(arr).repeat()
            padding = next(iter(temp_ds.batch(num_to_add)))
            if is_list_or_tuple:
                final_arr = type(arr)(tf.concat([a, p], 0) for a, p in zip(arr, padding))
            elif is_dict:
                final_arr = {k: tf.concat([arr[k], padding[k]], 0) for k in arr}
            else:
                final_arr = tf.concat([arr, padding], 0)
                
        elif remainder != 0:
            pad_needed = total_batch_size - remainder
            indices = tf.random.uniform(shape=[pad_needed], maxval=current_size, dtype=tf.int32)
            if is_list_or_tuple:
                padding = type(arr)(tf.gather(a, indices) for a in arr)
                final_arr = type(arr)(tf.concat([a, p], 0) for a, p in zip(arr, padding))
            elif is_dict:
                padding = {k: tf.gather(v, indices) for k, v in arr.items()}
                final_arr = {k: tf.concat([arr[k], padding[k]], 0) for k in arr}
            else:
                padding = tf.gather(arr, indices)
                final_arr = tf.concat([arr, padding], 0)

        # --- Dataset Creation ---
        dataset = tf.data.Dataset.from_tensor_slices(final_arr)
        if num_epochs != 1:
            dataset = dataset.repeat(count=num_epochs)
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        
        for batch_size in reversed(batch_dims_p):
            dataset = dataset.batch(batch_size, drop_remainder=False)
            
        return dataset.prefetch(tf.data.AUTOTUNE)
    
    # def make_dataset_p(arr, shuffle=False):   
    #     dataset = tf.data.Dataset.from_tensor_slices(arr)
    #     dataset_options = tf.data.Options()
    #     dataset_options.experimental_optimization.map_parallelization = True
    #     dataset_options.threading.private_threadpool_size = 48
    #     dataset_options.threading.max_intra_op_parallelism = 1
    #     # dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #     dataset = dataset.with_options(dataset_options)
    #     dataset = dataset.repeat(count=num_epochs)
    #     if shuffle:
    #         dataset = dataset.shuffle(shuffle_buffer_size)
    #     for batch_size in reversed(batch_dims_p):
    #         dataset = dataset.batch(batch_size, drop_remainder=True)
    #     return dataset.prefetch(prefetch_size)
                
        
    # def make_dataset_u(arr, shuffle=False):
#         dataset = tf.data.Dataset.from_tensor_slices(arr)
#         dataset_options = tf.data.Options()
#         dataset_options.experimental_optimization.map_parallelization = True
#         dataset_options.threading.private_threadpool_size = 48
#         dataset_options.threading.max_intra_op_parallelism = 1
#         # dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         dataset = dataset.with_options(dataset_options)
#         dataset = dataset.repeat(count=num_epochs)
#         if shuffle:
#             dataset = dataset.shuffle(shuffle_buffer_size)
#         for batch_size in reversed(batch_dims_u):
#             dataset = dataset.batch(batch_size, drop_remainder=True) 
#         return dataset.prefetch(prefetch_size)

    # Create tf.data.Datasets
    # xp_ds = tf.data.Dataset.from_tensor_slices(xp)
    # eval_ds = tf.data.Dataset.from_tensor_slices((xu_eval, yu_eval))
    # train_ds = tf.data.Dataset.from_tensor_slices((xu_train, yu_train))
    
    if config.train.mode in ["ul_batch", "ul_dataset"]:
        xp = xp.astype(np.float32)
        xp_ds = make_dataset_p((xp, yp))
        eval_ds = make_dataset_u((xu_eval, yu_eval))
        train_ds = make_dataset_p((xu_train, yu_train)) 
        
    if config.train.mode =='propensity':
        xp = xp.astype(np.float32)
        xp_ds = make_dataset_p((xp, yp))
        eval_ds = make_dataset_u((Xtest.astype(np.float32), ytest.astype(np.uint8)))
        train_ds = make_dataset_p((xu_train, yu_train)) 

    return xp_ds, train_ds, eval_ds, sz