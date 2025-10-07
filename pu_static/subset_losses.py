import jax
import jax.numpy as jnp
import math

def get_model_fn(model, params):
    def model_fn(x):
        return model.apply(params, x)
    return model_fn

def sq_Euclidean_cost(x, y):
    return jnp.linalg.norm(x-y, axis=tuple(range(1, x.ndim)))**2


def get_loss_T(config, model_T, model_eta):
    def loss_T(params_T, params_eta, data_batch, key):
       T_fn = get_model_fn(model_T, params_T)
       eta_fn = get_model_fn(model_eta, params_eta)
       x_0, _ = data_batch
       Tx0 = T_fn(x_0)
       
       loss = sq_Euclidean_cost(x_0, Tx0) + eta_fn(Tx0)
       
       translation = sq_Euclidean_cost(x_0, Tx0)
       
       return loss.mean(), translation
    return loss_T
   
def get_loss_eta(config, model_T, model_eta):
    c = config.c
    def loss_eta(params_T, params_eta, data_batch, key):
        T_fn = get_model_fn(model_T, params_T)
        eta_fn = get_model_fn(model_eta, params_eta)
        x_0, x_1 = data_batch
        Tx0 = T_fn(x_0)
        loss = - eta_fn(Tx0).mean() + c * jax.nn.relu(eta_fn(x_1)).mean()
        
        translation = sq_Euclidean_cost(x_0, Tx0)
        
        return loss, translation
    return loss_eta
        
        

# def sample_t(u0, n, t_0=0.0, t_1=1.0):
#     u = (u0 + math.sqrt(2)*jnp.arange(n*jax.device_count())) % 1
#     t = (t_1-t_0)*u[jax.process_index()*n:(jax.process_index()+1)*n] + t_0
#     return t[:, jnp.newaxis], u[-1]

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
#             x_t = x_t + wgf_step_size * jnp.clip(update, -1, 1) # wgf_step_size can multiplied with t. 
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