#代码示例：5-13
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Flatten, Dense, Input, Activation
from keras import Model
from keras.layers import GlobalAveragePooling2D

IMSIZE = 224
input_shape = (IMSIZE, IMSIZE, 3)
input_layer = Input(input_shape)
x = input_layer
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

x = BatchNormalization(axis=3)(x)
x = Conv2D(64, [3, 3], padding='same', activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Conv2D(64, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = BatchNormalization(axis=3)(x)
x = Conv2D(128, [3, 3], padding='same', activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Conv2D(128, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = BatchNormalization(axis=3)(x)
x = Conv2D(256, [3, 3], padding='same', activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Conv2D(256, [3, 3], padding='same', activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Conv2D(256, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = BatchNormalization(axis=3)(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = BatchNormalization(axis=3)(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = BatchNormalization(axis=3)(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(4096, activation = "relu")(x)
x = Dense(4096, activation = "relu")(x)
x = Dense(200, activation = "softmax")(x)
output_layer = x
model_vgg16 = Model(input_layer, output_layer)

model_vgg16_b = Model(input_layer, output_layer)
model_vgg16_b.summary()


# In[71]:


#代码示例：5-12
from keras.optimizers import Adam
model_vgg16.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

model_vgg16.fit_generator(train_generator,epochs=1,validation_data=validation_generator)


# In[4]:


#代码示例：5-15
from keras.preprocessing.image import ImageDataGenerator

IMSIZE=128

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'E:/why/CatDog/validation',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')

train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'E:/why/CatDog/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')


# In[5]:


#代码示例：5-16
import numpy as np
X,Y=next(validation_generator)
print(X.shape)
print(Y.shape)
Y[:,0]


# In[6]:


#代码示例：5-17
from matplotlib import pyplot as plt

plt.figure()
fig,ax = plt.subplots(2,5)
fig.set_figheight(6)
fig.set_figwidth(15)
ax=ax.flatten()
for i in range(10): ax[i].imshow(X[i,:,:,:])
plt.savefig('E:/why/plt save/test4.jpg')


# In[7]:


#代码示例：5-18
from keras.layers import Flatten,Input,BatchNormalization,Dense
from keras import Model
input_layer=Input([IMSIZE,IMSIZE,3])
x=input_layer
x=BatchNormalization()(x)
x=Flatten()(x)
x=Dense(2,activation='softmax')(x)
output_layer=x
model1=Model(input_layer,output_layer)
model1.summary()


# In[8]:


#代码示例：5-19
#限于时间只运行5个epoch
from keras.optimizers import Adam
model1.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.01),metrics=['accuracy'])
model1.fit_generator(train_generator,epochs=3,validation_data=validation_generator)


# In[ ]:


#代码示例：5-20
from keras.layers import Conv2D,MaxPooling2D

n_channel=100
input_layer=Input([IMSIZE,IMSIZE,3])
x=input_layer
x=BatchNormalization()(x)
x=Conv2D(n_channel,[2,2],activation='relu')(x)
x=MaxPooling2D([16,16])(x)
x=Flatten()(x)
x=Dense(2,activation='softmax')(x)
output_layer=x
model2=Model(input_layer,output_layer)
model2.summary()


# In[13]:


#代码示例：5-21
#限于时间只运行5个epoch
model2.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.01),metrics=['accuracy'])
model2.fit_generator(train_generator,epochs=2,validation_data=validation_generator)


# In[30]:


#代码示例：5-22
n_channel=20
input_layer=Input([IMSIZE,IMSIZE,3])
x=input_layer
x=BatchNormalization()(x)
for _ in range(7):
    x=Conv2D(n_channel,[2,2],padding='same',activation='relu')(x)
    x=MaxPooling2D([2,2])(x)
x=Flatten()(x)
x=Dense(2,activation='softmax')(x)
output_layer=x
model3=Model(input_layer,output_layer)
model3.summary()


# In[31]:


#代码示例：5-24
from keras.preprocessing.image import ImageDataGenerator

IMSIZE=128

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'E:/why/CatDog/validation',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')


# In[ ]:


#代码示例：5-23
#限于时间只运行5个epoch
model3.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.01),metrics=['accuracy'])
model3.fit_generator(train_generator,epochs=5,validation_data=validation_generator)


# In[33]:


train_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.5,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True).flow_from_directory(
    'E:/why/CatDog/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')


# In[34]:


#代码示例：5-26
from matplotlib import pyplot as plt

plt.figure()
fig,ax = plt.subplots(2,5)
fig.set_figheight(6)
fig.set_figwidth(15)
ax=ax.flatten()
X,Y=next(train_generator)
for i in range(10): ax[i].imshow(X[i,:,:,:])
plt.savefig('E:/why/plt save/test5.jpg')


# In[35]:


#代码示例：5-27
IMSIZE=128
from keras.layers import BatchNormalization,Conv2D,Dense,Flatten,Input,MaxPooling2D
from keras import Model

n_channel=100
input_layer=Input([IMSIZE,IMSIZE,3])
x=input_layer
x=BatchNormalization()(x)
for _ in range(7):
    x=BatchNormalization()(x)
    x=Conv2D(n_channel,[2,2],padding='same',activation='relu')(x)
    x=MaxPooling2D([2,2])(x)

x=Flatten()(x)
x=Dense(2,activation='softmax')(x)
output_layer=x
model=Model(input_layer,output_layer)
model.summary()


# In[36]:


#代码示例：5-28
#限于时间只运行5个epoch
from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001),metrics=['accuracy'])
model.fit_generator(train_generator,epochs=5,validation_data=validation_generator)
