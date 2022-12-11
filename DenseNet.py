#代码示例：6-11
from keras.preprocessing.image import ImageDataGenerator

IMSIZE=128

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'E:/why/data_des/test',
    target_size=(IMSIZE, IMSIZE),
    batch_size=100,
    class_mode='categorical')
train_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.5,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True).flow_from_directory(
    'E:/why/data_des/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size=100,
    class_mode='categorical')


# In[53]:


#代码示例：6-12
from matplotlib import pyplot as plt
plt.figure()
fig,ax = plt.subplots(2,5)
fig.set_figheight(6)
fig.set_figwidth(15)
ax=ax.flatten()
X,Y=next(train_generator)
for i in range(10): ax[i].imshow(X[i,:,:,:])
plt.savefig('E:/why/plt save/test6-3.jpg')

#代码示例：6-13
from keras.layers import Input,BatchNormalization,Conv2D,Dense,Flatten,Input,MaxPooling2D,Concatenate
from keras.layers.pooling import AveragePooling2D,GlobalAveragePooling2D
from keras import Model

IMSIZE=128

# 每个dense block中dense layer数目
nb_layers = 3
# 增长率
growth_rate = 32

# 输入层
input_layer = Input([IMSIZE,IMSIZE,3])
x = input_layer

# 初始卷积层  第一个 Feature block层
x = BatchNormalization()(x)
x = Conv2D(growth_rate*2, [7,7], strides=[2,2],padding='same', activation='relu')(x)
x=MaxPooling2D([3,3],strides=[2,2])(x)
x=BatchNormalization()(x)


# In[55]:


#代码示例：6-14  DENSENET 121层代码
# 设置 第一个Dense Block
for j in range(6):
    # 1.Dense Block
    # 用一个列表存放提供特征的层
    feature_list = [x]
    for i in range(nb_layers):
        x = BatchNormalization()(x)
        x = Conv2D(growth_rate, (1, 1), padding="same", activation='relu')(x)
        x = Conv2D(growth_rate, (3, 3), padding="same", activation='relu')(x)
        feature_list.append(x)
        if i<(nb_layers-1):
            x = Concatenate()(feature_list)


# In[56]:


# 2.设置 第一个Transition Block
x = BatchNormalization()(x)
x = Conv2D(growth_rate, (1, 1), padding="same", activation='relu')(x)
x = AveragePooling2D((2, 2), strides=(2, 2))(x)


# In[57]:


# 设置 第二个Dense Block
for j in range(12):
    # 1.Dense Block
    # 用一个列表存放提供特征的层
    feature_list = [x]
    for i in range(nb_layers):
        x = BatchNormalization()(x)
        x = Conv2D(growth_rate, (1, 1), padding="same", activation='relu')(x)
        x = Conv2D(growth_rate, (3, 3), padding="same", activation='relu')(x)
        feature_list.append(x)
        if i<(nb_layers-1):
            x = Concatenate()(feature_list)


# In[58]:


# 2.设置 第二个Transition Block
x = BatchNormalization()(x)
x = Conv2D(growth_rate, (1, 1), padding="same", activation='relu')(x)
x = AveragePooling2D((2, 2), strides=(2, 2))(x)


# In[59]:


# 设置 第三个Dense Block
for j in range(24):
    # 1.Dense Block
    # 用一个列表存放提供特征的层
    feature_list = [x]
    for i in range(nb_layers):
        x = BatchNormalization()(x)
        x = Conv2D(growth_rate, (1, 1), padding="same", activation='relu')(x)
        x = Conv2D(growth_rate, (3, 3), padding="same", activation='relu')(x)
        feature_list.append(x)
        if i<(nb_layers-1):
            x = Concatenate()(feature_list)


# In[60]:


# 2.设置 第三个Transition Block
x = BatchNormalization()(x)
x = Conv2D(growth_rate, (1, 1), padding="same", activation='relu')(x)
x = AveragePooling2D((2, 2), strides=(2, 2))(x)


# In[61]:


# 设置 第四个Dense Block
for j in range(16):
    # 1.Dense Block
    # 用一个列表存放提供特征的层
    feature_list = [x]
    for i in range(nb_layers):
        x = BatchNormalization()(x)
        x = Conv2D(growth_rate, (1, 1), padding="same", activation='relu')(x)
        x = Conv2D(growth_rate, (3, 3), padding="same", activation='relu')(x)
        feature_list.append(x)
        if i<(nb_layers-1):
            x = Concatenate()(feature_list)


# In[62]:


# 全局池化
x = GlobalAveragePooling2D()(x)
x = Dense(2,activation='softmax')(x)
output_layer = x
model = Model(input_layer,output_layer)
model.summary()


# In[63]:


#代码示例：6-16
from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.fit_generator(train_generator,epochs=5,validation_data=validation_generator)
