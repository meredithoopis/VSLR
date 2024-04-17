import jax
import flax
from flax import linen as nn
import jax.numpy as jnp
import time

from efficientnet import MBBlock
from transformer import TransformerBlock


class JAX_EfficientNet_SL(nn.Module):
    max_len : int = 64
    dim : int = 192
    kernel_size : int = 17
    dropout_rate : float = 0.3
    NUM_CLASSES : int = 250
    @nn.compact
    def __call__(self, inputs, deterministic=False, rng = None):
        x = nn.Dense(self.dim)(inputs)
        x = nn.BatchNorm(momentum = 0.95, use_running_average=deterministic)(x)
        
        config = {'deterministic': deterministic, 'rng': rng}
        
        x = MBBlock(self.dim, self.kernel_size)(x, **config)
        x = MBBlock(self.dim, self.kernel_size)(x, **config)
        x = MBBlock(self.dim, self.kernel_size)(x, **config)
        x = TransformerBlock(dim = self.dim)(x, **config)
        
        x = MBBlock(self.dim, self.kernel_size)(x, **config)
        x = MBBlock(self.dim, self.kernel_size)(x, **config)
        x = MBBlock(self.dim, self.kernel_size)(x, **config)
        x = TransformerBlock(dim = self.dim)(x, **config)
        
        x = nn.Dense(self.dim * 2)(x)
        x = nn.avg_pool(x, (x.shape[-2],)).squeeze(axis=-2)
        x = nn.Dropout(self.dropout_rate)(x, **config)
        x = nn.Dense(self.NUM_CLASSES)(x)
        return x
        # ...
        
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    input = jax.random.normal(key, shape=(32, 100, ))
    model = JAX_EfficientNet_SL()
    model.kernel_size = 7
    variables = model.init(key, input)
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    @jax.jit 
    def jit_model_apply(params, batch_stats ,x,rng): 
        return model.apply({'params':params,  'batch_stats':batch_stats}, x, deterministic= True, rng={'dropout': rng}) # mutable=['batch_stats']
    
    y = jit_model_apply(params,batch_stats,input,key) 
    
    start = time.time()
    y = jit_model_apply(params,batch_stats,input,key)  
    end = time.time()
    print('time for jax',end-start)
    
    flat_params, _ = jax.tree_util.tree_flatten(params)
    num_params = sum(p.size for p in flat_params)

    print(f"Number of learning parameters: {num_params}")
            