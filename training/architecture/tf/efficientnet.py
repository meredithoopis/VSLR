import tensorflow as tf 
import tensorflow.keras.layers as layers
import time
class ECA(layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)  #output channel 1, add zero padding 

    def call(self, inputs, mask=None):
        nn = layers.GlobalAveragePooling1D()(inputs, mask=mask)
        
        nn = tf.expand_dims(nn, -1)

        nn = self.conv(nn)

        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]

        return inputs * nn
    
class SE(layers.Layer): #Squeeze execution layer 
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)

class CausalDWConv1D(layers.Layer): #Causal depthwise convolution1D 
    def __init__(self, 
        kernel_size=15,
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer='glorot_uniform',
        name='', **kwargs):
        super().__init__(name=name,**kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(kernel_size-1),0),name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
                            kernel_size,
                            strides=1,
                            dilation_rate=dilation_rate,
                            padding='valid',
                            use_bias=use_bias,
                            depthwise_initializer=depthwise_initializer,
                            name=name + '_dwconv')
        self.supports_masking = True
        
    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x
    
    
def MBBlock(channel_size,
          kernel_size,
          dilation_rate=1,
          drop_rate=0.0,
          expand_ratio=2,
          activation='swish',
          name=None): #Mobile inverted bottleneck 

    if name is None:
        name = str(tf.keras.backend.get_uid("mblock"))
    # Expansion phase
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1] #Get number of channels 
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(
            channels_expand,
            use_bias=True,
            activation=activation,
            name=name + '_expand_conv')(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(kernel_size,
            dilation_rate=dilation_rate,
            use_bias=False,
            name=name + '_dwconv')(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

        x  = ECA()(x)

        x = tf.keras.layers.Dense(
            channel_size,
            use_bias=True,
            name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1), name=name + '_drop')(x)

        if (channels_in == channel_size):
            x = tf.keras.layers.add([x, skip], name=name + '_add') #Residual connection 
        return x

    return apply

if __name__ == "__main__":
    model = ECA()
    x = tf.random.normal((32, 100, 66))
    model(x)
    start = time.time()
    model(x)
    end = time.time()
    print('time for tf',end-start)