import jax
import flax
import jax.numpy as jnp 
from flax import linen as nn
from flax.linen import initializers 
from flax.linen.attention import dot_product_attention 
from jax import random 
from jax.lib import xla_bridge 



class SelfAttention(nn.Module): 
    dim: int   
    num_heads: int = 1 
    def setup(self):
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads" 
        self.head_dim = self.dim // self.num_heads 
        self.scale = self.head_dim ** 0.5 
        self.w_q = nn.Dense(self.dim)
        self.w_k = nn.Dense(self.dim)
        self.w_v = nn.Dense(self.dim)
    
    def __call__(self,x,padding_mask = None): 
        b, n, _ = x.shape 
        q = self.w_q(x).reshape((b,n, self.num_heads, self.head_dim)).transpose((0,2,1,3))
        k = self.w_k(x).reshape((b,n, self.num_heads, self.head_dim)).transpose((0,2,1,3))
        v = self.w_v(x).reshape((b,n, self.num_heads, self.head_dim)).transpose((0,2,1,3))
        if padding_mask is not None: 
            padding_mask = padding_mask[:, None, : , None]
        attn = dot_product_attention(q,k,v,bias=padding_mask, dropout_rate=0)
        attn = attn.transpose((0,2,1,3)).reshape((b,n,-1))
        return attn 
    
class TransformerBlock(nn.Module):
    dim: int
    num_heads: int = 1

    def setup(self):
        self.att = SelfAttention(self.dim, self.num_heads)
        #self.norm1 = nn.LayerNorm(self.dim)
        #self.norm2 = nn.LayerNorm(self.dim)
        self.norm1 = nn.LayerNorm(scale_init=initializers.ones, bias_init=initializers.zeros)
        self.norm2 = nn.LayerNorm(scale_init=initializers.ones, bias_init=initializers.zeros)
        self.dense1 = nn.Dense(self.dim*4)
        self.dense2 = nn.Dense(self.dim)
        self.dropout1 = nn.Dropout(rate=0.1)
        self.dropout2 = nn.Dropout(rate=0.2)
    
    def __call__(self, x, padding_mask=None, deterministic=False, rngs = None):
        if rngs is None: 
            rngs = {'dropout': self.make_rng('dropout')}
        dropout_rng = rngs['dropout']
        inputs = x
        x = self.att(self.norm1(x), padding_mask)
        x = self.dropout1(x, deterministic=deterministic, rng = dropout_rng)
        x = x + inputs

        att_out = x
        x = self.norm2(x)
        x = nn.gelu(self.dense1(x))
        x = self.dropout2(x, deterministic=deterministic, rng = dropout_rng)
        x = self.dense2(x)
        return att_out + x 


class Transformer(nn.Module):
    hidden_dim: int
    num_blocks: int
    num_heads: int = 1

    def setup(self):
        self.blocks = [TransformerBlock(self.hidden_dim, self.num_heads) for _ in range(self.num_blocks)]

    def __call__(self, x, padding_mask=None, deterministic = False, rngs = None):
        if rngs is None: 
            rngs = {'dropout': self.make_rng('dropout')}
        for block in self.blocks:
            x = block(x, padding_mask, deterministic, rngs)
        return x, None

if __name__ == "__main__": 
    print(jax.devices())
    print(xla_bridge.get_backend().platform) #Should return gpu 
    model = Transformer(hidden_dim=64, num_blocks=2, num_heads=8)

    rng = random.PRNGKey(0)
    x = random.normal(rng, (1, 10, 64))  #sqq length 10, hidden dim 64 
    variables = model.init(rng,x)
    params = variables['params']
    y, _ = model.apply({'params':params}, x, deterministic= True, rngs={'dropout': rng})

    print(y.shape)
    