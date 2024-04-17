import jax
import flax
from flax import linen as nn
import jax.numpy as jnp
import functools as ft
import time

    
class TransformerBlock(nn.Module):
    dim: int = 256
    num_head: int = 4
    dropout_rate = 0.1
    att_dropout_rate = 0.1
    
    @nn.compact
    def __call__(self, inputs, mask = None, training = True):
        x = inputs
        x = nn.LayerNorm(self.dim)(x)
        x = nn.SelfAttention(self.num_head, 
                            dropout_rate = self.att_dropout_rate,
                            deterministic=training
                            )(x)
        
        x = nn.Dropout(self.dropout_rate, deterministic = training)(x)
        x += inputs
        
        skip = x
        
        # FFN
        x = nn.LayerNorm(self.dim)(x)
        x = nn.gelu(nn.Dense(self.dim * 4)(x))
        x = nn.Dropout(self.dropout_rate, deterministic = training)(x)
        x = nn.Dense(self.dim)(x)
        
        x = nn.Dropout(self.dropout_rate, deterministic = training)(x)
        x += skip
        return x
        
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    input = jax.random.normal(key, shape=(32, 100, 256))
    model = TransformerBlock()
    model.kernel_size = 7
    variables = model.init(key, input)
    params = variables['params']
    
    
    @jax.jit 
    def jit_model_apply(params, x,rng): 
        return model.apply({'params':params}, x, deterministic = True, rng={'dropout': rng}) # mutable=['batch_stats']
    
    y = jit_model_apply(params,input,key) 
    
    start = time.time()
    y = jit_model_apply(params,input,key) 
    end = time.time()
    print('time for jax',end-start)    
        
