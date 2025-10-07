from functools import partial
import math

import jax
import jax.numpy as jnp
import flax
import optax
import diffrax
import numpy as np

from matplotlib import pyplot as plt


def get_eps_scheduler(config):
    return optax.piecewise_interpolate_schedule(interpolate_type='linear', init_value=1.0, boundaries_and_scales={1: 1.0, int(3*config.train.num_train_steps//4): 0.0})

def get_optimizer(config):
    return optax.adamw(learning_rate=config.lr)


# def get_optimizer(config):
#     optimizer = optax.adamw(learning_rate=config.lr)
#     optimizer = optax.chain(optax.clip(config.grad_clip), optimizer)
#     return optimizer

# def get_optimizer(config):
#     # scheduled_lr = optax.piecewise_constant_schedule(init_value=config.lr, boundaries_and_scales={5_000: 0.5, 10_000: 0.25, 15000: 0.125})
#     # optimizer = optax.adamw(learning_rate=scheduled_lr)
#     # optimizer = optax.chain(optax.clip(config.grad_clip), optimizer)
    
#     return optimizer
