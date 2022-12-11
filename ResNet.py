#代码示例：6-5
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

IMSIZE=224
train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'E:/why/data_res/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size=100,
    class_mode='categorical')

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'E:/why/data_res/validation',
    target_size=(IMSIZE, IMSIZE),
    batch_size=100,
    class_mode='categorical')

plt.figure()
fig,ax = plt.subplots(2,5)
fig.set_figheight(7)
fig.set_figwidth(15)
ax=ax.flatten()

X,Y=next(train_generator)
for i in range(10): ax[i].imshow(X[i,:,:,:])
plt.savefig('E:/why/plt save/test6_2.jpg')



#代码示例：6-6
from keras.layers import Input
from keras.layers import Activation, Conv2D, BatchNormalization, add, MaxPooling2D

NB_CLASS=3
IM_WIDTH=224
IM_HEIGHT=224

inpt = Input(shape=(IM_WIDTH, IM_HEIGHT, 3))

x = Conv2D(64, (7,7), padding='same', strides=(2,2), activation='relu')(inpt)
x = BatchNormalization()(x)
# 一个卷积层加一个batch normalization
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
x0 = x


# In[16]:


#代码示例：6-7
# 一个block
x = Conv2D(64, (1,1), padding='same', strides=(1,1), activation='relu')(x)
x = BatchNormalization()(x)
# 一个卷积层加一个batch normalization
x = Conv2D(64, (3,3), padding='same', strides=(1,1), activation='relu')(x)
x = BatchNormalization()(x)
# 一个卷积层加一个batch normalization
x = Conv2D(256, (1,1), padding='same', strides=(1,1), activation=None)(x)
x = BatchNormalization()(x)
# 一个卷积层加一个batch normalization

# 下面两步为了把输入64通道的数据转换为256个通道，用来让x0和x维数相同，可以进行加法计算，文章中虚线得的部分
x0 = Conv2D(256,(1,1),padding='same',strides=(1,1),activation='relu')(x0)
x0 = BatchNormalization()(x0)
x = add([x,x0])# add把输入的x和经过一个block之后输出的结果加在一起
x = Activation('relu')(x)#求和之后的结果再做一次relu
x0 = x # 把输入存到一个另外的变量中
#代码示例：6-7


# In[17]:


# 第二个block
x = Conv2D(64, (1,1), padding='same', strides=(1,1), activation='relu')(x)
x = BatchNormalization()(x)
# 一个卷积层加一个batch normalization
x = Conv2D(64, (3,3), padding='same', strides=(1,1), activation='relu')(x)
x = BatchNormalization()(x)
# 一个卷积层加一个batch normalization
x = Conv2D(256, (1,1), padding='same', strides=(1,1), activation=None)(x)
x = BatchNormalization()(x)
# 一个卷积层加一个batch normalization
add([x,x0])# add把输入的x和经过一个block之后输出的结果加在一起
x = Activation('relu')(x)#求和之后的结果再做一次relu
x0 = x # 把输入存到一个另外的变量中


# In[18]:


# 第三个block
x = Conv2D(64, (1,1), padding='same', strides=(1,1), activation='relu')(x)
x = BatchNormalization()(x)
# 一个卷积层加一个batch normalization
x = Conv2D(64, (3,3), padding='same', strides=(1,1), activation='relu')(x)
x = BatchNormalization()(x)
# 一个卷积层加一个batch normalization
x = Conv2D(256, (1,1), padding='same', strides=(1,1), activation=None)(x)
x = BatchNormalization()(x)
# 一个卷积层加一个batch normalization
add([x,x0])# add把输入的x和经过一个block之后输出的结果加在一起
x = Activation('relu')(x)#求和之后的结果再做一次relu
x0 = x # 把输入存到一个另外的变量中


# In[19]:


#代码示例：6-8
from keras.models import Model
model = Model(inputs=inpt,outputs=x)
model.summary()


# In[22]:


#代码示例：6-9
#在resnet最后的部分添加一个dense层，并输出一个二维的结果用来分类
from keras.layers import Dense, Flatten
x = model.output
x = Flatten()(x)
predictions = Dense(NB_CLASS,activation='softmax')(x)
model_res = Model(inputs=model.input,outputs=predictions)


# In[23]:


#代码示例：6-10
from keras.optimizers import Adam
model_res.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

model_res.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=10)