from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Dense, add
from keras.layers import GlobalAvgPool2D, GlobalMaxPool2D, concatenate

def adaptive_pooling(input):
    x_avg = GlobalAvgPool2D()(input)
    x_max = GlobalMaxPool2D()(input)
    return concatenate([x_avg, x_max])

def res_block(input, nf, n_blocks):
    x = conv_layer(input, nf)
    for _ in range(n_blocks): x = residual_layer(x, nf)
    return x

def residual_layer(input, nf):
    x = conv_layer(input, nf//2, ks=1, strides=(1,1), gamma_initializer='zeros')
    x = conv_layer(x, nf, strides=(1,1), gamma_initializer='zeros')
    return add([x, input])

def conv_layer(x, nf, ks=3, strides=(2,2), gamma_initializer='ones'):
    x = Conv2D(nf, ks, strides=strides, padding='same',
               activation='relu', kernel_initializer='he_uniform',
               use_bias=False)(x)
    x = BatchNormalization(gamma_initializer=gamma_initializer)(x)
    return x

def create_model(input_shape=(224,224,3), nf=16, blocks=[2,2,3,4], n_classes=6):
    m_in = Input(input_shape)
    x = conv_layer(m_in, nf)
    
    for block in blocks:
        nf *= 2
        x = res_block(x, nf, block)
    
    x = adaptive_pooling(x)
    m_out = Dense(n_classes, activation='softmax')(x)
    
    return Model(m_in, m_out)
