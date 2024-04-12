import jax 
import jax.numpy as jnp 
from flax import linen as nn 
from flax.linen import initializers 
from flax.linen.attention import make_causal_mask 

#https://flax.readthedocs.io/en/latest/_modules/flax/linen/attention.html


class ECA(nn.Module): #Efficient channel attention 
    kernel_size: int = 5 
    def setup(self): 
        self.conv = nn.Conv(features=1, kernel_size = self.kernel_size, strides = 1, padding="SAME", kernel_init = initializers.zeros)

    def __call__(self, inputs): 
        nn = jnp.mean(inputs,axis = 1, keepdims = True)
        nn = jnp.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = jnp.squeeze(nn, -1)
        nn = jax.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn 
    
class SE(nn.Module): 
    
    
