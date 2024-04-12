import jax 
import jax.numpy as jnp 
from flax import linen as nn 
from flax.linen import initializers 
from flax.linen.attention import make_causal_mask 
import optax 
from jax import random 
from flax.training import train_state 
import time 


#https://flax.readthedocs.io/en/latest/_modules/flax/linen/attention.html


class ECA(nn.Module): #Efficient channel attention 
    kernel_size: int = 5 
    def setup(self): 
        self.conv = nn.Conv(features=1, kernel_size = (1,self.kernel_size), strides = (1,1), padding="SAME")

    def __call__(self, inputs): 
        nn = jnp.mean(inputs,axis = 1, keepdims = True)
        #nn = jnp.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = jnp.squeeze(nn, axis = [1,2])
        nn = jax.nn.sigmoid(nn)
        return inputs * nn[:, None, None]
    
class SE(nn.Module): 
    reduction_ratio: int = 16 
    def setup(self): 
        self.avg_pool = nn.avg_pool 
        self.fc1 = nn.DenseGeneral
        self.fc2 = nn.DenseGeneral
    
    def __call__(self, x): 
        bs, h, w, channels = x.shape 
        #se = jnp.mean(x,axis = (1,2), keepdims=True)
        se = self.avg_pool(x, [h, w])

        se = self.fc1(se, features = channels // self.reduction_ratio)
        se = jax.nn.relu(se)
        se = self.fc2(se, features = channels)
        se = jax.nn.sigmoid(se)
        return x * se[:, None, :]

class CausalDWConv1D(nn.Module): #Causal depth wise convolution 
    kernel_size: int = 15 
    dilation_rate: int = 1 
    use_bias : bool = False 
    depthwise_initializer: initializers.Initializer = initializers.glorot_uniform()
    def setup(self): 
        self.dw_conv = nn.Conv(features =1 , kernel_size = self.kernel_size, strides=1, padding="VALID", kernel_init = self.depthwise_initializer)

    def __call__(self, inputs): 
        pad_w = ((0,0), (self.dilation_rate * (self.kernel_size -1), 0), (0,0))
        x = jnp.pad(inputs, pad_w)
        x = self.dw_conv(x)
        return x 
    
def MBBlock(channel_size, kernel_size, dilation_rate = 1, 
            drop_rate = 0, expand_ratio= 2, 
            activation = jax.nn.swish, name=None): 
    class MBBlock(nn.Module): 
        def setup(self): 
            self.dense1 = nn.Dense(features=channel_size * expand_ratio, kernel_init= initializers.zeros, use_bias = True) 
            self.dwconv = CausalDWConv1D(kernel_size = kernel_size, dilation_rate = dilation_rate)
            self.norm = nn.BatchNorm(momentum = 0.95)
            self.eca = ECA()
            self.dense2 = nn.Dense(features = channel_size, kernel_init = initializers.zeros, use_bias = True)
            self.dropout = nn.Dropout(rate = drop_rate)

        def __call__(self, inputs): 
            channels_in = inputs.shape[-1]
            channels_expand = channels_in * expand_ratio 
            skip = inputs 
            x = self.dense1(inputs)
            x = activation(x)
            x = self.dwconv(x)
            x = self.norm(x)
            x = self.eca(x)
            x = self.dense2(x)
            if drop_rate > 0: 
                x = self.dropout(x)
            if channels_in == channel_size: 
                x = x + skip 
            return x 
    return MBBlock 


if __name__ == "__main__": 
    key = random.PRNGKey(0) #pseudo-random number generator (PRNG) key given an integer seed
    x = random.normal(key, (32,100,66))
    model = ECA()
    params = model.init(key, x)
    start = time.time()
    y = model.apply(params,x)
    end = time.time()
    print(y.shape)
    print('Time for jax', end-start)