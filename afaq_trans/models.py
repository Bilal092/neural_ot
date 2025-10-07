import jax
import jax.numpy as jnp
import jax.random as random
import flax
from flax import linen as nn
from typing import Any, Callable, Sequence, Tuple
import math



@flax.struct.dataclass
class State:
  step: int
  opt_state: Any
  model_params: Any
  ema_rate: Any
  params_ema: Any
  key: Any
  c: Any
  
  
# def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
#   # print("t_shape", timesteps.shape)
#   assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
#   half_dim = embedding_dim // 2
#   # magic number 10000 is from transformers
#   emb = math.log(max_positions) / (half_dim - 1)
#   # emb = math.log(2.) / (half_dim - 1)
#   emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
#   # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
#   # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
#   emb = timesteps[:, None] * emb[None, :]
#   emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
#   if embedding_dim % 2 == 1:  # zero pad
#     emb = jnp.pad(emb, [[0, 0], [0, 1]])
#   assert emb.shape == (timesteps.shape[0], embedding_dim)
#   return emb


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



# Due to diffrax shape compatability issues we created two model archietctures one for evluation and other for training.
class Smodel_eval(nn.Module):
  num_hid : int
  num_out : int 
  t_embed_dim: int

  @nn.compact
  def __call__(self, t, x):
    
    if jnp.ndim(t) == 0:
        t = jnp.broadcast_to(t, x.shape[0:-1]+(1,))
        
    t_embed = get_timestep_embedding(t.squeeze(), self.t_embed_dim)
      
    h = jnp.concatenate([t_embed,x], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.hstack([t,h])
    h = jnp.concatenate([t_embed,h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.hstack([t,h])
    h = jnp.concatenate([t_embed,h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.hstack([t,h])
    h = jnp.concatenate([t_embed,h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.hstack([t,h])
    # h = jnp.concatenate([t,h], axis=-1)
    h = nn.Dense(self.num_out)(h)
    return h


class Smodel(nn.Module):
   num_hid : int
   num_out : int
   t_embed_dim: int

   @nn.compact
   def __call__(self, t, x):
    if t.ndim == 0:  # Scalar input
         t = jnp.broadcast_to(t, (x.shape[0], 1, 1)) 
    

    #  t_embed = get_timestep_embedding(t.squeeze(), self.t_embed_dim)
    t_embed = get_timestep_embedding(t, self.t_embed_dim)
    print(t_embed.shape)
    t_embed = jnp.reshape(t_embed, x.shape[0:-1]+(self.t_embed_dim,))
    h = jnp.concatenate([t_embed, x], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.hstack([t,h])
    h = jnp.concatenate([t_embed,h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.hstack([t,h])
    h = jnp.concatenate([t_embed,h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.hstack([t,h])
    h = jnp.concatenate([t_embed,h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.hstack([t,h])
    # h = jnp.concatenate([t,h], axis=-1)
    h = nn.Dense(self.num_out)(h)
    return h
  
  
class Qmodel(nn.Module):
  num_hid : int
  num_out : int
  t_embed_dim: int

  @nn.compact
  def __call__(self, t, x_0, x_1):        
    # h = jnp.hstack([t, x_0, x_1, t<0.5]) # removed the heaviside 
    # h = jnp.hstack([t, x_0, x_1]) jnp.linspace(0.0, 1.0, 10).reshape((1,-1))
    
    t_embed = get_timestep_embedding(t.squeeze(), self.t_embed_dim)
    h = jnp.concatenate([t_embed, x_0, x_1, t<0.5], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.hstack([t, h])
    h = jnp.concatenate([t_embed, h], axis=-1)  
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.hstack([t, h])
    h = jnp.concatenate([t_embed, h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.hstack([t, h])
    h = jnp.concatenate([t_embed, h], axis=-1)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    # h = jnp.hstack([t, h])
    #h = jnp.concatenate([t, h], axis=-1)
    h = nn.Dense(self.num_out)(h)
    # out = (1-t)*x_0 + t*(x_1 + h)
    
    out = (1-t)*x_0 + t*(x_1 + h)
    
    return out


# class Qmodel(nn.Module):
#   num_hid : int
#   num_out : int
#   t_embed_dim: int

#   @nn.compact
#   def __call__(self, t, x_0, x_1):        
#     # h = jnp.hstack([t, x_0, x_1, t<0.5]) # removed the heaviside 
#     # h = jnp.hstack([t, x_0, x_1]) jnp.linspace(0.0, 1.0, 10).reshape((1,-1))
     
#     def transport_net(input):
#       MLP_out = nn.Sequential([
#         nn.Dense(self.num_hid),
#         nn.swish,
#         nn.Dense(self.num_hid),
#         nn.swish,
#         nn.Dense(self.num_hid),
#         nn.swish,
#         nn.Dense(self.num_hid),
#         nn.swish,
#         nn.Dense(self.num_out),])(input)
#       # ResConnect = nn.Dense(self.num_out)(input)
#       ResConnect = input
#       return MLP_out + ResConnect
    
#     x_1p = transport_net(x_1)
    
#     t_embed = get_timestep_embedding(t.squeeze(), self.t_embed_dim)
#     h = jnp.concatenate([t_embed, x_0, x_1p, t<0.5], axis=-1)
#     h = nn.Dense(self.num_hid)(h)
#     h = nn.swish(h)
#     # h = jnp.hstack([t, h])
#     h = jnp.concatenate([t_embed, h], axis=-1)  
#     h = nn.Dense(self.num_hid)(h)
#     h = nn.swish(h)
#     # h = jnp.hstack([t, h])
#     h = jnp.concatenate([t_embed, h], axis=-1)
#     h = nn.Dense(self.num_hid)(h)
#     h = nn.swish(h)
#     # h = jnp.hstack([t, h])
#     h = jnp.concatenate([t_embed, h], axis=-1)
#     h = nn.Dense(self.num_hid)(h)
#     h = nn.swish(h)
#     # h = jnp.hstack([t, h])
#     #h = jnp.concatenate([t, h], axis=-1)
#     h = nn.Dense(self.num_out)(h)
#     # out = (1-t)*x_0 + t*(x_1 + h)
    
    
#     out = (1-t)*x_0 + t*x_1p + t*(1-t)*h 
    
#     return out








