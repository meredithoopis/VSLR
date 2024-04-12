from efficientnet import MBBlock
from transformer import TransformerBlock
from flax import linen as nn
from flax.linen import initializers
from jax import random
import time
import jax.numpy as jnp

CHANNELS = 66
PAD = -100
NUM_CLASSES = 250

class Model(nn.Module):
    max_len: int = 64
    dim: int = 192

    def setup(self):
        self.dense1 = nn.Dense(features=self.dim, kernel_init=initializers.zeros, use_bias=False, name='stem_conv')
        self.norm = nn.BatchNorm(momentum=0.95, name='stem_bn')
        self.mbblock1 = MBBlock(channel_size=self.dim, kernel_size=17, drop_rate=0.2)
        self.mbblock2 = MBBlock(channel_size=self.dim, kernel_size=17, drop_rate=0.2)
        self.mbblock3 = MBBlock(channel_size=self.dim, kernel_size=17, drop_rate=0.2)
        self.transformer = TransformerBlock(dim=self.dim, num_heads=2)
        self.mbblock4 = MBBlock(channel_size=self.dim, kernel_size=17, drop_rate=0.2)
        self.mbblock5 = MBBlock(channel_size=self.dim, kernel_size=17, drop_rate=0.2)
        self.mbblock6 = MBBlock(channel_size=self.dim, kernel_size=17, drop_rate=0.2)
        self.transformer2 = TransformerBlock(dim=self.dim, num_heads=2)
        self.dense2 = nn.Dense(features=self.dim*2, name='top_conv')
        self.dropout = nn.Dropout(rate=0.3)
        self.dense3 = nn.Dense(features=NUM_CLASSES, name='classifier')
        #self.dropout_rng = self.make_rng

    def __call__(self, inputs, deterministic=False, rngs=None):
        if rngs is None: 
            rngs = {'dropout': self.make_rng('dropout')}
        x = self.dense1(inputs)
        x = self.norm(x, use_running_average=deterministic)
        x = self.mbblock1(x, rng=self.make_rng('dropout'))
        x = self.mbblock2(x, rng=self.make_rng('dropout'))
        x = self.mbblock3(x, rng=self.make_rng('dropout'))
        x = self.transformer(x, rngs= rngs)
        x = self.mbblock4(x, rng=self.make_rng('dropout'))
        x = self.mbblock5(x, rng=self.make_rng('dropout'))
        x = self.mbblock6(x, rng=self.make_rng('dropout'))
        x = self.transformer2(x, rngs=rngs)
        x = self.dense2(x)
        x = jnp.mean(x, axis=1)
        x = self.dropout(x, deterministic=deterministic, rng=self.make_rng('dropout'))
        x = self.dense3(x)
        return x

if __name__ == "__main__":
    key = random.PRNGKey(0)
    x = random.normal(key, (32, 64, 66))
    model = Model()
    variables = model.init({'params': key}, x)
    params = variables['params']
    batch_stats = variables['batch_stats']
    y = model.apply({'params': params, 'batch_stats': batch_stats}, x, rngs={'dropout': random.PRNGKey(0)}, mutable=['batch_stats'])
    start = time.time()
    y, updated_stats = model.apply({'params': params, 'batch_stats': batch_stats}, x, rngs={'dropout': random.PRNGKey(0)}, mutable=['batch_stats'])
    end = time.time()
    print("JAX model time: ", end-start)
    print(y.shape)