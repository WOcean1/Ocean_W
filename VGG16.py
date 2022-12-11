#代码示例：5-9
from keras.preprocessing.image import ImageDataGenerator

IMSIZE = 224

train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
   'E:/why/data_vgg/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size=32,
    class_mode='categorical')

validation_generator = ImageDataGenerator(
    rescale=1. / 255).flow_from_directory(
        'E:/why/data_vgg/test',
        target_size=(IMSIZE, IMSIZE),
        batch_size=32,
        class_mode='categorical')


# In[67]:


#代码示例：5-10
from matplotlib import pyplot as plt

plt.figure()
fig, ax = plt.subplots(2, 5)
fig.set_figheight(6)
fig.set_figwidth(15)
ax = ax.flatten()
X, Y = next(validation_generator)
for i in range(10):
    ax[i].imshow(X[i, :, :, ])
plt.savefig('E:/why/plt save/test3.jpg')


# In[68]:


#代码示例：5-11(书稿的印刷中疏忽了全连接层)
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Input, Activation
from keras import Model
from keras.layers import GlobalAveragePooling2D

IMSIZE = 224
input_shape = (IMSIZE, IMSIZE, 3)
input_layer = Input(input_shape)
x = input_layer

x = Conv2D(64, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(64, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(128, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(256, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(256, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(4096, activation = "relu")(x)
x = Dense(4096, activation = "relu")(x)
x = Dense(200, activation = "softmax")(x)
output_layer = x
model_vgg16 = Model(input_layer, output_layer)
model_vgg16.summary()


# In[69]:


#代码示例：5-12
from keras.optimizers import Adam
model_vgg16.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model_vgg16.fit_generator(train_generator,epochs=1,validation_data=validation_generator)

