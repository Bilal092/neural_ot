import jax
from jax import numpy as jnp
from flax import linen as nn
import math
'''
etamodel: neural network parameterization for eta function
Tmodel: neural network parameterization for T function
'''
class etamodel(nn.Module):
  num_hid : int
  num_out : int
  
  @nn.compact
  def __call__(self, x):

    h = nn.Dense(self.num_hid)(x)
    h = nn.swish(h)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    h = nn.Dense(self.num_hid)(h)
    h = nn.swish(h)
    h = nn.Dense(self.num_out)(h)
    return h

class Tmodel(nn.Module):
  num_hid : int
  num_out : int
    
  @nn.compact
  def __call__(self, x):
    def transport_net(x):
        MLP_out = nn.Sequential([
          nn.Dense(self.num_hid),
          nn.swish,
          nn.Dense(self.num_hid),
          nn.swish,
          nn.Dense(self.num_hid),
          nn.swish,
          nn.Dense(self.num_hid),
          nn.swish,
          nn.Dense(self.num_out),])(x)
        ResConnect = nn.Dense(self.num_out)(x)
        return MLP_out + ResConnect
      
    output = transport_net(x)
    return output

