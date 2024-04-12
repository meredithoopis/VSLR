from efficientnet import MBBlock
from transformer import TransformerBlock
import tensorflow as tf
import time

CHANNELS = 66
PAD = -100
NUM_CLASSES = 250


def get_model(max_len=64, dim=192 ):
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

    x = tf.keras.layers.Dense(dim*2,activation=None,name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES,name='classifier')(x)
    return tf.keras.Model(inp, x)

if __name__ == "__main__":
    x = tf.random.normal((32, 64, 66))
    model = get_model()
    model.summary()
    y = model(x)
    start = time.time()
    y = model(x)
    end = time.time()
    print("TF model time: ", end-start)
    # print(y.shape)