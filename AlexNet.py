#代码示例：5-5
from keras.preprocessing.image import ImageDataGenerator

IMSIZE=227

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'E:/why/ChineseStyle/test/',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')

train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'E:/why/ChineseStyle/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')


# In[63]:


#代码示例：5-6
from matplotlib import pyplot as plt

plt.figure()
fig,ax = plt.subplots(2,5)
fig.set_figheight(7)
fig.set_figwidth(15)
ax=ax.flatten()
X,Y=next(validation_generator)
for i in range(10): ax[i].imshow(X[i,:,:,:])
plt.savefig('E:/why/plt save/test2.jpg')


# In[64]:


#代码示例：5-7
from keras.layers import Activation,Conv2D, BatchNormalization, Dense
from keras.layers import Dropout, Flatten, Input, MaxPooling2D, ZeroPadding2D
from keras import Model

IMSIZE = 227
input_layer = Input([IMSIZE,IMSIZE,3])
x = input_layer
x = Conv2D(96,[11,11],strides = [4,4], activation = 'relu')(x)
x = MaxPooling2D([3,3], strides = [2,2])(x)
x = Conv2D(256,[5,5],padding = "same", activation = 'relu')(x)
x = MaxPooling2D([3,3], strides = [2,2])(x)
x = Conv2D(384,[3,3],padding = "same", activation = 'relu')(x)
x = Conv2D(384,[3,3],padding = "same", activation = 'relu')(x)
x = Conv2D(256,[3,3],padding = "same", activation = 'relu')(x)
x = MaxPooling2D([3,3], strides = [2,2])(x)
x = Flatten()(x)
x = Dense(4096,activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096,activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(2,activation = 'softmax')(x)
output_layer=x
model=Model(input_layer,output_layer)
model.summary()


# In[65]:


#代码示例：5-8
from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.fit_generator(train_generator,epochs=1,validation_data=validation_generator)
