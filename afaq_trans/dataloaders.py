import jax

from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import pandas as pd
import math



    
def get_dataset(dataset, batch_size=64, additional_dim=None, source=False, evaluation=False):
    
    num_devices = jax.local_device_count()
    per_device_batch_size = batch_size // num_devices
    shuffle_buffer_size = 100000
    prefetch_size = tf.data.experimental.AUTOTUNE
    # num_epochs = None if not evaluation else 1
    num_epochs = 1
    
    # Create additional data dimension when jitting multiple steps together 
    # we are jitting one step but with larger memory one can jit multiple batches together. 
    
    if additional_dim is None:
        batch_dims = [jax.local_device_count(), per_device_batch_size]
    else:
        batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size]
    
    
    if  dataset in ["ffhq_MAN", "ffhq_WOMAN", "ffhq_ADULT", "ffhq_YOUNG"]:
        train_size = 60000
        test_size = 10000
        latents = np.load("/neural-ot-ss/lagrangian-pot-flows/subset_selection_embeddings/data/latents.npy")  # ALAE embeddings
        gender = np.load("/neural-ot-ss/lagrangian-pot-flows/subset_selection_embeddings/data/gender.npy")  # male/female labels
        age = np.load("/neural-ot-ss/lagrangian-pot-flows/subset_selection_embeddings/data/age.npy") # age labels

        train_latents, test_latents = latents[:train_size], latents[train_size:]
        train_gender, test_gender = gender[:train_size], gender[train_size:]
        train_age, test_age = age[:train_size], age[train_size:]
        
        if dataset == "ffhq_MAN":
            inds_train = np.arange(train_size)[(train_gender == "male").reshape(-1)]
            inds_test = np.arange(test_size)[(test_gender == "male").reshape(-1)]
        
        elif dataset == "ffhq_WOMAN":
            inds_train = np.arange(train_size)[(train_gender == "female").reshape(-1)]
            inds_test = np.arange(test_size)[(test_gender == "female").reshape(-1)]
            
        elif dataset == "ffhq_ADULT":
            inds_train = np.arange(train_size)[(train_age > 44).reshape(-1)*(train_age != -1).reshape(-1)]
            inds_test = np.arange(test_size)[(test_age > 44).reshape(-1)*(test_age != -1).reshape(-1)]
            
        elif dataset == "ffhq_YOUNG":
            inds_train = np.arange(train_size)[((train_age > 16) & (train_age <= 44)).reshape(-1)*(train_age != -1).reshape(-1)]
            inds_test = np.arange(test_size)[((test_age > 16) & (test_age <= 44)).reshape(-1)*(test_age != -1).reshape(-1)]
            
        data_train = train_latents[inds_train]
        data_test = test_latents[inds_test]
        
        if source == True:
            data_test_gender = test_gender[inds_test]
            data_test_age = test_age[inds_test]
            
        def create_dataset(arr, validation=False):
            dataset = tf.data.Dataset.from_tensor_slices(arr)
            dataset_options = tf.data.Options()
            dataset_options.experimental_optimization.map_parallelization = True
            dataset_options.threading.private_threadpool_size = 48
            dataset_options.threading.max_intra_op_parallelism = 1
            # dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.with_options(dataset_options)
            dataset = dataset.repeat(count=num_epochs)
            dataset = dataset.shuffle(shuffle_buffer_size)
            for batch_size in reversed(batch_dims):
                dataset = dataset.batch(batch_size, drop_remainder=True)
            return dataset.prefetch(prefetch_size)
        
        train_ds = create_dataset((data_train))
        if source == True:
            test_ds = create_dataset((data_test, data_test_gender, data_test_age), validation=True)
        else:
            test_ds = create_dataset((data_test), validation=True)
        
        return train_ds, test_ds 
    
    
class PhiFilteredDS:
    def __init__(self, config, base_dataset, phi_fn,  phi_model, params, eps_scheduler, threshold=1e-5,
                 evaluation=False, additional_dim=None, uniform_dequantization=False):
        self.config = config
        self.base_dataset = base_dataset
        self.phi_model = phi_model
        self.phi_fn = phi_fn
        self.params = params  # already replicated across devices
        self.threshold = threshold
        self.eps_scheduler = eps_scheduler
        self.epsilon = 1.0
        # self.batch_size = config.train.batch_size
        self.batch_size = config.train.batch_size
        self.refresh_every = config.phi_sampler.refresh_every
        self.min_acceptance_ratio = config.phi_sampler.min_acceptance_ratio  # this can set from confuguration file outside
        self.candidate_batch_size = config.phi_sampler.candidate_batch_size
        self.evaluation = evaluation
        self.additional_dim = config.train.n_jitted_steps
        self.uniform_dequantization = config.data.uniform_dequantization # self to False fo tabular data and True for images
        self.step = 0
        self.dataset = self.base_dataset
        
        # self.temprature = 1 / 20

        if self.batch_size % jax.device_count() != 0:
            raise ValueError(f'Batch size ({self.batch_size}) must be divisible by number of devices ({jax.device_count()})')

        self.per_device_batch_size = self.batch_size // jax.device_count()
        self.candidate_per_device_batch_size = self.candidate_batch_size // jax.device_count()
        self.shuffle_buffer_size = 10000
        self.prefetch_size = tf.data.experimental.AUTOTUNE
        self.num_epochs = 1

        # Expected shape: (num_devices, num_jit_steps, per_device_batch_size, ...)
        if self.additional_dim is None:
            self.batch_dims = [jax.local_device_count(), self.per_device_batch_size]
            self.candidate_batch_dims = [jax.local_device_count(), self.candidate_per_device_batch_size]
        else:
            self.batch_dims = [jax.local_device_count(), self.additional_dim, self.per_device_batch_size]
            self.candidate_batch_dims = [jax.local_device_count(), self.additional_dim, self.candidate_per_device_batch_size]    
        
    def _evaluate_phi(self, x_batch):
        """Evaluate φ(x) with already-sharded input and replicated params."""
        if isinstance(x_batch, dict):
            x_batch = x_batch['image']

        if hasattr(x_batch, "numpy"):
            x_np = x_batch.numpy()
        else:
            x_np = tf.convert_to_tensor(x_batch).numpy()

        # phi_vals = self.phi_fn(self.params, jnp.ones(x_np.shape[:-1] + (1,)), x_np)
        # print("x_np.shape", x_np.shape)
        phi_vals = self.phi_fn(self.params, x_np)

        return jnp.squeeze(phi_vals, axis=-1)

    def create_dataset_with_options(self, dataset, batch_dims, shuffle=False):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.threading.private_threadpool_size = 48
        dataset_options.threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(dataset_options)
        dataset = dataset.repeat(count=self.num_epochs)

        # print(f"length of dataset_inside 1={len(dataset)}")
        
        if shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_size)

        for dim in reversed(batch_dims):
            dataset = dataset.batch(dim, drop_remainder=True)

        #print(f"length of dataset_inside 2 ={len(dataset)}")
        #print(f"length of prefetched base_dataset_inside={len(dataset.prefetch(self.prefetch_size))}")
        return dataset.prefetch(self.prefetch_size)

    # helpers once in your class
    def _flatten_leading(self, t):
        t = tf.convert_to_tensor(t)
        nd = len(self.batch_dims)                 # e.g. 2 or 3 (D, S?, B)
        s  = tf.shape(t)
        n  = tf.reduce_prod(s[:nd])               # N = D*(S?)*B
        return tf.reshape(t, tf.concat([[n], s[nd:]], axis=0))

    def _refresh_dataset(self):
        print(f"[INFO] Refreshing dataset at step {self.step}...")
    
        # Evaluate φ on convenient candidate batches (unbatched -> candidate_batch_dims)
        print(f"length of base_dataset outside={len(self.base_dataset)}")
        
        # candidate_ds = self.create_dataset_with_options(self.base_dataset.unbatch(), self.candidate_batch_dims)

        candidate_ds = self.base_dataset # start with assigning base dataset to candidate dataset 
        if self.batch_dims!=self.candidate_batch_dims: 
            for dim in self.batch_dims:
                candidate_ds = candidate_ds.unbatch()
            unbatched_base_ds = candidate_ds
            for dim in reversed(self.candidate_batch_dims):
                candidate_ds = candidate_ds.batch(dim, drop_remainder=True)
        else:
            unbatched_base_ds = self.base_dataset
            for dim in self.batch_dims:
                unbatched_base_ds = unbatched_base_ds.unbatch()
            
        print(f"length of rebatched base_dataset outside={len(candidate_ds)}")

        print(f"length of candidate_ds={len(candidate_ds)}")
    
        # accepted_x, accepted_y = [], []
    
        # for batch in candidate_ds:
        #     if isinstance(batch, tuple):
        #         x_batch, y_batch = batch
        #     else:
        #         x_batch, y_batch = batch, None
    
        #     phi_vals = self._evaluate_phi(x_batch)    # shape compatible with x_batch
        #     mask = phi_vals <= self.threshold
    
        #     accepted_x.append(tf.boolean_mask(x_batch, mask))
        #     if y_batch is not None:
        #         accepted_y.append(tf.boolean_mask(y_batch, mask))


        # figure out how many leading batch axes the *candidate* stream has
        n_leading = len(getattr(self, "candidate_batch_dims", self.batch_dims))
        accepted_x, accepted_y = [], []
        
        for batch in candidate_ds:
            # normalize batch to (x_batch, y_batch or None) without changing your style
            if isinstance(batch, dict):
                if "image" in batch and "label" in batch:
                    x_batch, y_batch = batch["image"], batch["label"]
                elif "x" in batch and "y" in batch:
                    x_batch, y_batch = batch["x"], batch["y"]
                elif "image" in batch:
                    x_batch, y_batch = batch["image"], None
                else:
                    # fallback: first tensor-like is x
                    k = sorted(batch.keys())[0]
                    x_batch, y_batch = batch[k], None
            elif isinstance(batch, tuple):
                x_batch, y_batch = batch
            else:
                x_batch, y_batch = batch, None
        
            # ---- φ(x): returns np/jnp; squeeze trailing 1 if present; flatten leading dims -> [N]
            phi_vals = self._evaluate_phi(x_batch)  # shape like [*, ..., 1?]
            phi_vals = np.asarray(phi_vals)
            if phi_vals.ndim > n_leading and phi_vals.shape[-1] == 1:
                phi_vals = phi_vals[..., 0]
            phi_flat = phi_vals.reshape(-1)  # [N]
        
            # ---- TF bool mask over N examples
            mask = tf.convert_to_tensor(phi_flat <= self.threshold, dtype=tf.bool)  # [N]
        
            # ---- flatten x (and y) to [N, ...] so we can boolean_mask along axis 0
            x_tf = tf.convert_to_tensor(x_batch)
            sx = tf.shape(x_tf)
            N  = tf.reduce_prod(sx[:n_leading])
            x_flat = tf.reshape(x_tf, tf.concat([[N], sx[n_leading:]], axis=0))
            accepted_x.append(tf.boolean_mask(x_flat, mask))
        
            if y_batch is not None:
                y_tf = tf.convert_to_tensor(y_batch)
                sy = tf.shape(y_tf)
                Ny = tf.reduce_prod(sy[:n_leading])
                y_flat = tf.reshape(y_tf, tf.concat([[Ny], sy[n_leading:]], axis=0))
                accepted_y.append(tf.boolean_mask(y_flat, mask))

    
        # If nothing passed the mask, just fall back to original (not an "acceptance ratio" decision;
        # this avoids concat([]) crash).
        if not accepted_x:
            print("[WARN] No φ-accepted samples; using original dataset only.")
            # self.dataset = self.create_dataset_with_options(self.base_dataset.unbatch(), self.batch_dims)
            # print(f"No φ-accepted samples; length of mixed dataset {len(self.dataset)}")
            self.dataset = self.base_dataset
            print(f"No φ-accepted samples; length of mixed dataset {len(self.dataset)}")
            return
    
        # Build φ-pool tensors
        x_phi = tf.concat(accepted_x, axis=0)
        y_phi = tf.concat(accepted_y, axis=0) if accepted_y else None
    
        # Trim to match batch shape product so both pipelines batch identically
        required = int(np.prod(self.batch_dims))
        n_samples = int(x_phi.shape[0])
        remainder = n_samples % required
        if remainder > 0:
            print(f"[INFO] Trimming {remainder} φ-samples to match batch shape {self.batch_dims}")
            x_phi = x_phi[:-remainder]
            if y_phi is not None:
                y_phi = y_phi[:-remainder]
    
        # φ-dataset with the same structure and batching as original
        if y_phi is None:
            unbatched_phi_dataset = tf.data.Dataset.from_tensor_slices(x_phi)
        else:
            unbatched_phi_dataset = tf.data.Dataset.from_tensor_slices((x_phi, y_phi))
        
        # Original dataset, rebatched to ensure identical element_spec
        # orig_dataset = self.create_dataset_with_options(self.base_dataset.unbatch(), self.batch_dims)
        
        # Affine mixture: (1-e)·φ + e·orig
        w_phi = 1.0 - float(self.epsilon)
        w_orig = float(self.epsilon)
        mixed_dataset = tf.data.Dataset.sample_from_datasets(
            datasets=[unbatched_phi_dataset, unbatched_base_ds],
            weights=[w_phi, w_orig],
            stop_on_empty_dataset=False,
        )
    
        self.dataset = self.create_dataset_with_options(mixed_dataset, self.batch_dims)

        print(f"[INFO] Using affine mixture (1-e)·φ + e·orig with α={self.epsilon:.3f}")
    

    def _make_dataset(self, x, y):
        if y is None:
            ds = tf.data.Dataset.from_tensor_slices(x)
        else:
            ds = tf.data.Dataset.from_tensor_slices((x, y))
        return self.create_dataset_with_options(ds, self.batch_dims)

    def update_if_needed(self, step, new_params=None):
        self.step = step
        if step % self.refresh_every == 0 and step > 0:
            if new_params is not None:
                self.params = new_params
            self.epsilon = self.eps_scheduler(step)
            self._refresh_dataset()
            
    def create_batch_iterator(self):
        iterator = iter(self.dataset.repeat())

        def batch_iterator(key=None):
            nonlocal iterator  # needed to reassign iterator after StopIteration

            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.dataset)
                batch = next(iterator)

            # Handle case: batch is dict with 'image' key
            if isinstance(batch, dict) and 'image' in batch:
                batch = batch['image']

            # Handle case: batch is a tuple (x, y)
            if isinstance(batch, tuple):
                x, y = batch
                if hasattr(x, '_numpy'):
                    x = x._numpy()
                else:
                    x = tf.convert_to_tensor(x).numpy()

                if hasattr(y, '_numpy'):
                    y = y._numpy()
                else:
                    y = tf.convert_to_tensor(y).numpy()

                return (x, y)

            # Handle case: single tensor
            if hasattr(batch, '_numpy'):
                return batch._numpy()
            else:
                return tf.convert_to_tensor(batch).numpy()

        return batch_iterator


    def get_dataset(self):
        return self.dataset
    
    
# class PhiFilteredDS:
#   def __init__(self, config, base_dataset, phi_model, params,
#                 evaluation=False, additional_dim=None, uniform_dequantization=False):
#       """
#       Parameters:
#       - base_dataset: tf.data.Dataset yielding (x,) or (x, y)
#       - phi_model: Flax model (Module)
#       - params: Current parameters of φ
#       - batch_size: final training batch size
#       - refresh_every: how often to refresh filtering
#       - min_acceptance_ratio: fallback to full data if acceptance too low
#       - candidate_batch_size: batch size for φ evaluation
#       """
#       self.config = config
#       self.base_dataset = base_dataset
#       self.phi_model = phi_model
#       self.params = params
#       self.batch_size = config.data.batch_size
#       self.refresh_every = config.phi_sampler.refresh_every
#       self.min_acceptance_ratio = config.phi_sampler.min_acceptance_ratio
#       self.candidate_batch_size = config.phi_sampler.candidate_batch_size
#       self.evaluation = evaluation
#       self.additional_dim = config.train.n_jitted_steps
#       self.uniform_dequantization = config.data.uniform_dequantization
      
      
#       self.step = 0
#       self.dataset = self.base_dataset
#       self.temprature = 1/20
      
      
#       if self.batch_size % jax.device_count() != 0:
#         raise ValueError(f'Batch sizes ({self.batch_size} must be divided by'
#                         f'the number of devices ({jax.device_count()})')

#       self.per_device_batch_size = self.batch_size // jax.device_count()
#       # Reduce this when image resolution is too large and data pointer is stored
#       self.shuffle_buffer_size = 10000
#       self.prefetch_size = tf.data.experimental.AUTOTUNE
#       self.num_epochs = None if not self.evaluation else 1
#       # Create additional data dimension when jitting multiple steps together
#       if self.additional_dim is None:
#         self.batch_dims = [jax.local_device_count(), self.per_device_batch_size]
#       else:
#         self.batch_dims = [jax.local_device_count(), self.additional_dim, self.per_device_batch_size]

#         # self._refresh_dataset()

#   # def _evaluate_phi(self, x_batch):
#   #     """Runs φ(x) on a batch of TensorFlow tensors using Flax model on GPU."""
#   #     # x_batch_jax = jax.device_put(tf.convert_to_tensor(x_batch).numpy())  # to JAX array
#   #     x_batch_jax = jnp.array(x_batch)
#   #     phi_vals = self.phi_model.apply(self.params, jnp.ones((len(x_batch_jax), 1)), x_batch_jax).squeeze()
#   #     return jax.device_get(phi_vals)  # return as NumPy array for filtering
  
#   def _evaluate_phi(self, x_batch):
#     """Evaluate φ(x) on a batch which may be a tensor or a dict with 'image' key."""
#     # Extract image tensor
#     if isinstance(x_batch, dict):
#         x_batch = x_batch['image']
    
#     # Convert to NumPy and then JAX array
#     if hasattr(x_batch, "numpy"):
#         x_batch_np = x_batch.numpy()
#     else:
#         x_batch_np = tf.convert_to_tensor(x_batch).numpy()

#     x_batch_jax = jnp.array(x_batch_np)

#     # Run φ model
#     phi_vals = self.phi_model.apply(
#         self.params,
#         jnp.ones(x_batch_jax.shape[:-1]+(1,)),
#         x_batch_jax
#     ).squeeze()

#     return jax.device_get(phi_vals)

  
#   def create_dataset_with_options(self, dataset):
#     dataset_options = tf.data.Options()
#     # Set dataset options
#     dataset_options.experimental_optimization.map_parallelization = True
#     dataset_options.threading.private_threadpool_size = 48
#     dataset_options.threading.max_intra_op_parallelism = 1
#     dataset = dataset.with_options(dataset_options)
#     dataset = dataset.repeat(count=self.num_epochs)
#     # dataset = dataset.shuffle(shuffle_buffer_size)
#     for batch_size in reversed(self.batch_dims):
#       dataset = dataset.batch(batch_size, drop_remainder=True)
#     return dataset.prefetch(self.prefetch_size)

#   def _refresh_dataset(self):
#       print(f"[INFO] Refreshing dataset at step {self.step}...")
#       candidate_ds = self.base_dataset.unbatch().batch(self.candidate_batch_size)

#       accepted_x, accepted_y = [], []
#       total, accepted = 0, 0

#       for batch in candidate_ds:
#           if isinstance(batch, tuple):
#               x_batch, y_batch = batch
#           else:
#               x_batch, y_batch = batch, None

#           phi_vals = self._evaluate_phi(x_batch)
#           mask = phi_vals <= 1.0e-5 # slightly positive thresholding is done instead of having exact zero.

#           total += len(phi_vals)
#           accepted += jnp.sum(mask)

#           accepted_x.append(tf.boolean_mask(x_batch, mask))
#           if y_batch is not None:
#               accepted_y.append(tf.boolean_mask(y_batch, mask))

#       acceptance_ratio = accepted / total if total > 0 else 0
#       # print(f"[INFO] φ-based acceptance rate: {acceptance_ratio:.2%}")

#       if acceptance_ratio < self.min_acceptance_ratio:
#           # print("[WARN] Too few accepted — using full dataset.")
#           # self.dataset = self._make_dataset_from_tfds(self.base_dataset)
#           self.dataset = self.dataset
#       else:
#           x_all = tf.concat(accepted_x, axis=0)
#           y_all = tf.concat(accepted_y, axis=0) if accepted_y else None
#           self.dataset = self._make_dataset(x_all, y_all)

#   def _make_dataset(self, x, y):
#     if y is None:
#       ds = tf.data.Dataset.from_tensor_slices(x)
#       ds = self.create_dataset_with_options(ds)
#     else:
#       ds = tf.data.Dataset.from_tensor_slices((x, y))
#       ds = self.create_dataset_with_options(ds)
#     return ds

#   def _make_dataset_from_tfds(self, ds):
#       # return ds.shuffle(10_000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
#       return self.create_dataset_with_options(ds)

#   def update_if_needed(self, step, new_params=None):
#       self.step = step
#       if step % self.refresh_every == 0 and step>0:
#           if new_params is not None:
#               self.params = new_params
#           self._refresh_dataset()
          
#   def create_batch_iterator(self):
#     iterator = iter(self.dataset)
#     def batch_iterator(key=None):  # `key` is unused but kept for consistency
#         batch = next(iterator)
#         if isinstance(batch, dict) and 'image' in batch:
#             batch = batch['image']
#         return batch._numpy()
#     return batch_iterator
    
#   def get_dataset(self):
#       return self.dataset         
            
            
            
        
        
        
        
        
            
        
            
        
        
            
            
            
        
        



