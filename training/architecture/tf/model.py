from .efficientnet import MBBlock
from .transformer import TransformerBlock

CHANNELS = 66

def get_model(max_len=64, dropout_step=0, dim=192, ):
    inp = tf.keras.Input((max_len,CHANNELS))
    x = tf.keras.layers.Masking(mask_value=PAD,input_shape=(max_len,CHANNELS))(inp)
    ksize = 17
    x = tf.keras.layers.Dense(dim, use_bias=False,name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)

    x = MBBlock(dim,ksize,drop_rate=0.2)(x)
    x = MBBlock(dim,ksize,drop_rate=0.2)(x)
    x = MBBlock(dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(dim,expand=2)(x)

    x = MBBlock(dim,ksize,drop_rate=0.2)(x)
    x = MBBlock(dim,ksize,drop_rate=0.2)(x)
    x = MBBlock(dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(dim,expand=2)(x)

    if dim == 384: #for the 4x sized model
        x = MBBlock(dim,ksize,drop_rate=0.2)(x)
        x = MBBlock(dim,ksize,drop_rate=0.2)(x)
        x = MBBlock(dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(dim,expand=2)(x)

        x = MBBlock(dim,ksize,drop_rate=0.2)(x)
        x = MBBlock(dim,ksize,drop_rate=0.2)(x)
        x = MBBlock(dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(dim,expand=2)(x)

    x = tf.keras.layers.Dense(dim*2,activation=None,name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES,name='classifier')(x)
    return tf.keras.Model(inp, x)