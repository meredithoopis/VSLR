import jax 
import jax.numpy as jnp 
from flax import linen as nn 
from flax.linen import initializers 

class ECA(nn.Module):

    kernel_size = 5
        
    @nn.compact 
    # @ft.partial(jax.jit, donate_argnums=0, static_argnums=1) 
    def __call__(self, inputs, mask=None):
        embed = inputs.shape[-2]
        length = inputs.shape[-1]
        x = nn.avg_pool(inputs, (embed,))
        x = jnp.reshape(x, (-1, length, 1))

        x = nn.Conv(1, kernel_size=(self.kernel_size,), strides=1, padding="SAME", use_bias=False) (x)

        x = nn.sigmoid(x)
        x = jnp.reshape(x,(-1, 1, length))
        return inputs * x
    
    
class CausalDWConv1D(nn.Module): #Causal depth wise convolution 
    kernel_size: int = 15 
    dilation_rate: int = 1 
    use_bias : bool = False 
    depthwise_initializer: initializers.Initializer = initializers.glorot_uniform()

    @nn.compact
    def __call__(self, inputs): 
        features = inputs.shape[-1]
        x  = nn.Conv(features = features, 
                                    kernel_size = (self.kernel_size,), strides=1, 
                                    padding=self.kernel_size//2, 
                                    kernel_init = self.depthwise_initializer,
                                    feature_group_count = features)(inputs)
        return x 
    
class MBBlock(nn.Module):
    
    output_dim: int
    kernel_size: int 
    ratio: int = 2
    stride: int = 1
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training = True):
        inputs = x
        input_dim = x.shape[-1]
        hidden_dim = input_dim * self.ratio
        x = nn.Dense(hidden_dim)(x)
        x = nn.silu(x)
        
        x = CausalDWConv1D(kernel_size=self.kernel_size)(x)
        x = nn.BatchNorm(momentum = 0.95, use_running_average = not training)(x)
        
        x = ECA()(x)

            
        x = nn.Dense(self.output_dim)(x)

        if self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate, deterministic = training)(x)
        
        if input_dim == self.output_dim:
            x = x + inputs    
        
        return x