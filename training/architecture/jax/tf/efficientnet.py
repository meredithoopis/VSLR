import jax
import flax
from flax import linen as nn
import jax.numpy as jnp
import functools as ft
import time
class ECA(nn.Module):

    kernel_size = 5
        
    @nn.compact 
    # @ft.partial(jax.jit, donate_argnums=0, static_argnums=1) 
    def __call__(self, inputs, mask=None):
        embed = inputs.shape[-2]
        length = input.shape[-1]
        bs = input.shape[0]
        x = nn.avg_pool(inputs, (embed,))
        x = jnp.reshape(x, (bs, length, 1))

        x = nn.Conv(1, kernel_size=(self.kernel_size,), strides=1, padding="SAME", use_bias=False) (x)

        x = nn.sigmoid(x)
        x = jnp.reshape(x,(bs, 1, length))
        return inputs * x
    
# class CausalDWConv1D(nn.Module):
#     def __init__(self, 
#         kernel_size=15,
#         dilation_rate=1,
#         use_bias=False,
#         depthwise_initializer='glorot_uniform',
#         name='', **kwargs):
#         super().__init__(name=name,**kwargs)
#         self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(kernel_size-1),0),name=name + '_pad')
#         self.dw_conv = tf.keras.layers.DepthwiseConv1D(
#                             kernel_size,
#                             strides=1,
#                             dilation_rate=dilation_rate,
#                             padding='valid',
#                             use_bias=use_bias,
#                             depthwise_initializer=depthwise_initializer,
#                             name=name + '_dwconv')
#         self.supports_masking = True
        
#     def call(self, inputs):
#         x = self.causal_pad(inputs)
#         x = self.dw_conv(x)
#         return x
    
    
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    input = jax.random.normal(key, shape=(32, 100, 66))
    model = ECA()
    params = model.init(key, input)
    y = model.apply(params, input)
    start = time.time()
    model.apply(params, input)
    end = time.time()
    print('time for jax',end-start)