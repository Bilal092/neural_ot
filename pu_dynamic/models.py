import jax
from jax import numpy as jnp
from flax import linen as nn
import math

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Flatten to 1D regardless of shape
    timesteps = jnp.reshape(timesteps, (-1,))
    
    half_dim = embedding_dim // 2
    exponent = -math.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * exponent)
    emb = timesteps[:, None] * emb[None, :]  # (B, half_dim)
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)  # (B, embedding_dim)

    if embedding_dim % 2 == 1:
        emb = jnp.pad(emb, [(0, 0), (0, 1)])  # pad final dim

    return emb

class Smodel(nn.Module):
  num_hid : int
  num_out : int

  @nn.compact
  def __call__(self, t, x):
    if jnp.ndim(t) == 0:
        t = jnp.broadcast_to(t, x.shape[0:-1]+(1,))
    # print("s_t.shape", t.shape)
    # print("s_x.shape", x.shape)
    h = jnp.concatenate([t,x], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.concatenate([t,h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.concatenate([t,h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.concatenate([t,h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.concatenate([t,h], axis=-1)
    h = nn.Dense(self.num_out)(h)
    return h
  
  
  
class Qmodel(nn.Module):
  num_hid : int
  num_out : int
    
  @nn.compact
  def __call__(self, t, x_0, x_1):
    
    # h = jnp.concatenate([t, x_0, x_1p, t<0.5], axis=-1) # removed the heaviside 
    # h = jnp.hstack([t, x_0, x_1]) jnp.linspace(0.0, 1.0, 10).reshape((1,-1))
    h = jnp.concatenate([t, x_0, x_1, t<0.5], axis=-1) # removed the heaviside
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.concatenate([t, h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.concatenate([t, h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.concatenate([t, h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.concatenate([t, h], axis=-1)
    h = nn.Dense(self.num_out)(h)
    
    x_t = (1-t)*x_0 + t*(x_1) + t*(1-t)*h

    return x_t

# class Qmodel(nn.Module):
#   num_hid : int
#   num_out : int
    
#   @nn.compact
#   def __call__(self, t, x_0, x_1):
#     def transport_net(inputs):
#       MLP_out = nn.Sequential([
#             nn.Dense(self.num_hid),
#             nn.swish,
#             nn.Dense(self.num_hid),
#             nn.swish,
#             nn.Dense(self.num_hid),
#             nn.swish,
#             nn.Dense(self.num_hid),
#             nn.swish,
#             nn.Dense(self.num_out),
#         ])(inputs)
#       #ResConnect = nn.Dense(self.num_out)(inputs)
#       ResConnect = inputs
#       return MLP_out + ResConnect

#     # print("x1_shape", x_1.shape)
#     x_1p = transport_net(x_1)
      
#     # h = jnp.concatenate([t, x_0, x_1p, t<0.5], axis=-1) # removed the heaviside 
#     # h = jnp.hstack([t, x_0, x_1]) jnp.linspace(0.0, 1.0, 10).reshape((1,-1))
#     h = jnp.concatenate([t, x_0, x_1p], axis=-1) # removed the heaviside
#     h = nn.Dense(self.num_hid)(h)
#     h = nn.swish(h)
#     # h = jnp.concatenate([t, h], axis=-1)
#     h = nn.Dense(self.num_hid)(h)
#     h = nn.swish(h)
#     # h = jnp.concatenate([t, h], axis=-1)
#     h = nn.Dense(self.num_hid)(h)
#     h = nn.swish(h)
#     # h = jnp.concatenate([t, h], axis=-1)
#     h = nn.Dense(self.num_hid)(h)
#     h = nn.swish(h)
#     # h = jnp.concatenate([t, h], axis=-1)
#     h = nn.Dense(self.num_out)(h)

#     return x_1p, h

# class Smodel(nn.Module):
#     config: ml_collections.ConfigDict
#     num_hid: int
#     num_out: int
#     t_embed_dim: int

#     @nn.compact
#     def __call__(self, t, x):
#         # Ensure t has shape (..., 1)
#         if t.ndim < x.ndim:
#             t = jnp.broadcast_to(t, x.shape[:-1] + (1,))

#         # Get embedding
#         t_flat = t.reshape(-1)
#         t_emb = get_timestep_embedding(t_flat, self.t_embed_dim)
#         t_embed = t_emb.reshape(t.shape[:-1] + (self.t_embed_dim,))

#         h = jnp.concatenate([t_embed, x], axis=-1)
#         h = nn.Dense(self.num_hid)(h)
#         h = nn.swish(h)
#         h = nn.Dense(self.num_hid)(h)
#         h = nn.swish(h)
#         h = nn.Dense(self.num_hid)(h)
#         h = nn.swish(h)
#         out = nn.Dense(self.num_out)(h)
#         return out


# class Qmodel(nn.Module):
#     config: ml_collections.ConfigDict
#     num_hid: int
#     num_out: int
#     t_embed_dim: int

#     @nn.compact
#     def __call__(self, t, x_0, x_1):
#         if t.ndim < x_0.ndim:
#             t = jnp.broadcast_to(t, x_0.shape[:-1] + (1,))

#         # Time embedding
#         t_flat = t.reshape(-1)
#         t_emb = get_timestep_embedding(t_flat, self.t_embed_dim)
#         t_embed = t_emb.reshape(t.shape[:-1] + (self.t_embed_dim,))

#         # Optional flag
#         t_flag = (t < 0.5).astype(jnp.float32)
#         print("t_embed.shape", t_embed.shape)
#         print("x_0.shape", x_0.shape)

#         # Joint input
#         h = jnp.concatenate([t_embed, x_0, x_1, t_flag], axis=-1)
#         h = nn.Dense(self.num_hid)(h)
#         h = nn.swish(h)
#         h = nn.Dense(self.num_hid)(h)
#         h = nn.swish(h)
#         h = nn.Dense(self.num_hid)(h)
#         h = nn.swish(h)
#         h_out = nn.Dense(self.num_out)(h)

#         # Transport network
#         def transport_net(input):
#             mlp = nn.Sequential([
#                 nn.Dense(self.num_hid), nn.swish,
#                 nn.Dense(self.num_hid), nn.swish,
#                 nn.Dense(self.num_hid), nn.swish,
#                 nn.Dense(self.num_hid), nn.swish,
#                 nn.Dense(self.num_out)
#             ])
#             res = nn.Dense(self.num_out)(input)
#             return mlp(input) + res

#         x_1p = transport_net(x_1)

#         return x_1p, h_out