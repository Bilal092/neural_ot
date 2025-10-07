import jax
import jax.numpy as jnp
import math

def get_model_fn(model, params):
    def model_fn(t, x):
        return model.apply(params, t, x)
    return model_fn


def sample_t(u0, n, t_0=0.0, t_1=1.0):
    u = (u0 + math.sqrt(2)*jnp.arange(n*jax.device_count())) % 1
    t = (t_1-t_0)*u[jax.process_index()*n:(jax.process_index()+1)*n] + t_0
    return t[:, jnp.newaxis], u[-1]


# loss functions with rejection sampling based interpolant

def get_loss_s(config, model_s, model_q):
    # returns  loss function for potential
    c = config.c 
    wgf_steps = config.train.wgf_steps
    wgf_step_size = config.train.wgf_step_size
    def loss_s(params_s, params_q, data_batch, key):
        s = get_model_fn(model_s, params_s)
        dsdtdx_fn = jax.grad(lambda t, x: s(t, x).sum(), argnums=[0, 1])
        def potential(t, x):
            dsdt, dsdx = dsdtdx_fn(t, x)
            return dsdt + 0.5 * (dsdx**2).sum(1, keepdims=True)
        acceleration_fn = jax.grad(lambda t, x: potential(t, x).sum(), argnums=1)
        u0 = jax.random.uniform(key, 1)
        t, _ = sample_t(u0, data_batch[0].shape[0])
        t_0, t_1 = jnp.zeros([len(t), 1]), jnp.ones([len(t), 1])
        x_0, x_1, x_1t = data_batch[0], data_batch[1], data_batch[2]  # data_batch[2] comes from the rejection sampling
        # print("s_x_0.shape", x_0.shape)
        # print("s_x_1.shape", x_1.shape)
        # print("s_t.shape", t.shape)
        # x_1p, h = model_q.apply(params_q, t, x_0, x_1t)  # rejection sampled input for interpolant 
        x_t = model_q.apply(params_q, t, x_0, x_1t)  # rejection sampled input for interpolant
        
        # print("s_x_t.shape", x_t.shape)
        for _ in range(wgf_steps):
            update = jax.lax.stop_gradient(acceleration_fn(t, x_t))
            x_t = x_t + wgf_step_size * t * jnp.clip(update, -1, 1) # wgf_step_size can multiplied with t.
             
        x_t = jax.lax.stop_gradient(x_t)
        loss = c * jax.nn.relu(-s(t_1, x_1)) + s(t_0, x_0)
        loss += potential(t, x_t).mean()
        update = jax.lax.stop_gradient(acceleration_fn(t, x_t))
        return loss.mean(), jnp.sqrt((update**2).sum(1)).mean()
    return loss_s

def get_loss_q(config, model_s, model_q):
    # returns  loss function for interpolant
    c = config.c 
    wgf_steps = config.train.wgf_steps
    wgf_step_size = config.train.wgf_step_size
    def loss_q(params_s, params_q, data_batch, key):
        s_stopped = get_model_fn(model_s, jax.lax.stop_gradient(params_s))
        dsdtdx_fn_stopped = jax.grad(lambda t, x: s_stopped(t, x).sum(), argnums=[0, 1])
        def potential_stopped(t, x):
            dsdt, dsdx = dsdtdx_fn_stopped(t, x)
            return dsdt + 0.5 * (dsdx**2).sum(1, keepdims=True)
        acceleration_fn = jax.grad(lambda t, x: potential_stopped(t, x).sum(), argnums=1)
        u0 = jax.random.uniform(key, 1)
        t, _ = sample_t(u0, data_batch[0].shape[0])
        t_0, t_1 = jnp.zeros([len(t), 1]), jnp.ones([len(t), 1])
        x_0, x_1t = data_batch[0], data_batch[2]
        # x_1p, h = model_q.apply(params_q, t, x_0, x_1t)
        # x_t = (1 - t) * x_0 + t * x_1p + t * (1 - t) * h
        x_t = model_q.apply(params_q, t, x_0, x_1t)
        update = jax.lax.stop_gradient(acceleration_fn(t, x_t))
        loss = -potential_stopped(t, x_t)
        update = jax.lax.stop_gradient(acceleration_fn(t, x_t))
        return loss.mean(), jnp.sqrt((update**2).sum(1)).mean()
    return loss_q



# def get_loss_s(config, model_s, model_q):
#     # returns  loss function for potential
#     c = config.c 
#     wgf_steps = config.train.wgf_steps
#     wgf_step_size = config.train.wgf_step_size
#     def loss_s(params_s, params_q, data_batch, key):
#         s = get_model_fn(model_s, params_s)
#         dsdtdx_fn = jax.grad(lambda t, x: s(t, x).sum(), argnums=[0, 1])
#         def potential(t, x):
#             dsdt, dsdx = dsdtdx_fn(t, x)
#             return dsdt + 0.5 * (dsdx**2).sum(1, keepdims=True)
#         acceleration_fn = jax.grad(lambda t, x: potential(t, x).sum(), argnums=1)
#         u0 = jax.random.uniform(key, 1)
#         t, _ = sample_t(u0, data_batch[0].shape[0])
#         t_0, t_1 = jnp.zeros([len(t), 1]), jnp.ones([len(t), 1])
#         x_0, x_1 = data_batch[0], data_batch[1]
#         # print("s_x_0.shape", x_0.shape)
#         # print("s_x_1.shape", x_1.shape)
#         # print("s_t.shape", t.shape)
#         x_1p, h = model_q.apply(params_q, t, x_0, x_1)
#         x_t = (1 - t) * x_0 + t * x_1p + t * (1 - t) * h
#         # x_t = (1-t)*x_0 + t*(x_1 + h)
#         # print("s_x_t.shape", x_t.shape)
#         for _ in range(wgf_steps):
#             update = jax.lax.stop_gradient(acceleration_fn(t, x_t))
#             x_t = x_t + wgf_step_size * t * jnp.clip(update, -1, 1) # wgf_step_size can multiplied with t. 
#         x_t = jax.lax.stop_gradient(x_t)
#         loss = c * jax.nn.relu(-s(t_1, x_1)) + s(t_0, x_0)
#         loss += potential(t, x_t).mean()
#         update = jax.lax.stop_gradient(acceleration_fn(t, x_t))
#         return loss.mean(), jnp.sqrt((update**2).sum(1)).mean()
#     return loss_s

# def get_loss_q(config, model_s, model_q):
#     # returns  loss function for interpolant
#     c = config.c 
#     wgf_steps = config.train.wgf_steps
#     wgf_step_size = config.train.wgf_step_size
#     def loss_q(params_s, params_q, data_batch, key):
#         s_stopped = get_model_fn(model_s, jax.lax.stop_gradient(params_s))
#         dsdtdx_fn_stopped = jax.grad(lambda t, x: s_stopped(t, x).sum(), argnums=[0, 1])
#         def potential_stopped(t, x):
#             dsdt, dsdx = dsdtdx_fn_stopped(t, x)
#             return dsdt + 0.5 * (dsdx**2).sum(1, keepdims=True)
#         acceleration_fn = jax.grad(lambda t, x: potential_stopped(t, x).sum(), argnums=1)
#         u0 = jax.random.uniform(key, 1)
#         t, _ = sample_t(u0, data_batch[0].shape[0])
#         t_0, t_1 = jnp.zeros([len(t), 1]), jnp.ones([len(t), 1])
#         x_0, x_1 = data_batch[0], data_batch[1]
#         x_1p, h = model_q.apply(params_q, t, x_0, x_1)
#         x_t = (1 - t) * x_0 + t * x_1p + t * (1 - t) * h
#         # x_t = (1-t)*x_0 + t*(x_1 + h)
#         update = jax.lax.stop_gradient(acceleration_fn(t, x_t))
#         loss = -potential_stopped(t, x_t)
#         update = jax.lax.stop_gradient(acceleration_fn(t, x_t))
#         return loss.mean(), jnp.sqrt((update**2).sum(1)).mean()
#     return loss_q