#代码示例：6-17
from keras.preprocessing.image import ImageDataGenerator
import random
random.seed(2019425)

# Set image size
IMSIZE=112

# laod validation data
datagen = ImageDataGenerator(rescale=1. / 255,
                             shear_range=0.5,
                             rotation_range=30,
                             zoom_range=0.2,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True,
                             validation_split = 0.4)

validation_generator = datagen.flow_from_directory(
    'E:/why/data_mob/',
    target_size=(IMSIZE, IMSIZE),
    batch_size=10,
    class_mode='categorical',
    subset = 'validation')

train_generator = datagen.flow_from_directory(
    'E:/why/data_mob/',
    target_size=(IMSIZE, IMSIZE),
    batch_size=20,
    class_mode='categorical',
    subset = 'training')

from matplotlib import pyplot as plt
plt.ion()
plt.figure()
fig, ax = plt.subplots(2, 5)
fig.set_figheight(6)
fig.set_figwidth(15)
ax = ax.flatten()
X, Y = next(validation_generator)
for i in range(10):
    ax[i].imshow(X[i, :, :, ])
plt.savefig('E:/why/plt save/test6-4.jpg')
plt.show()


# 代码示例：6-18
def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    # Arguments

        inputs: Input tensor of shape `(rows, cols, channels)`

        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).

        alpha: controls the width of the network.

        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.

        block_id: Integer, a unique identification designating
            the block number.

    # Returns
        Output tensor of block.

    """

    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = ZeroPadding2D(((0, 1), (0, 1)),
                          name='conv_pad_%d' % block_id)(inputs)

    x = DepthwiseConv2D((3, 3),
                        padding='same' if strides == (1, 1) else 'valid',
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=-1,
                           name='conv_dw_%d_bn' % block_id)(x)
    x = ReLU(6., name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=-1,
                           name='conv_pw_%d_bn' % block_id)(x)

    return ReLU(6., name='conv_pw_%d_relu' % block_id)(x)


#代码示例：6-19
from keras.optimizers import Adam
from keras.layers import ZeroPadding2D, ReLU, DepthwiseConv2D
from keras.layers import Input
from keras.layers import Input,BatchNormalization,Conv2D,Dense,Flatten,Input,MaxPooling2D,Concatenate
from keras.layers.pooling import AveragePooling2D,GlobalAveragePooling2D
from keras import Model
alpha = 1
depth_multiplier = 1
# 输入层
input_layer = Input([IMSIZE,IMSIZE,3])

# 初始卷积层
x = input_layer
x = ZeroPadding2D(padding = ((0,1),(0,1)),name='conv1_pad')(x)
x = Conv2D(32,(3,3),padding='valid',use_bias=False,strides=(2,2),name='conv1')(x)
x = BatchNormalization(axis=-1, name='conv1_bn')(x)
x = ReLU(6,name='conv1_relu')(x)

# 保留其中的一些深度可分离卷积层
x = _depthwise_conv_block(x, 64, alpha, block_id=1)
x = _depthwise_conv_block(x, 128, alpha, strides=(2, 2), block_id=2)
x = _depthwise_conv_block(x, 256, alpha, strides=(2, 2), block_id=3)
x = _depthwise_conv_block(x, 512, alpha, strides=(2, 2), block_id=4)
x = _depthwise_conv_block(x, 1024, alpha, strides=(2, 2), block_id=5)


# Average Pooling
x = GlobalAveragePooling2D()(x)

x = Dense(10,activation='softmax')(x)
model = Model(inputs=input_layer,outputs=x)
model.summary()


#代码示例：6-20
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=50)