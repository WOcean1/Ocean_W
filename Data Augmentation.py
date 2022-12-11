#代码示例：5-25
#代码示例：5-24
from keras.preprocessing.image import ImageDataGenerator

IMSIZE=128

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'E:/why/CatDog/validation',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')
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

#代码示例：5-28
#限于时间只运行5个epoch
from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001),metrics=['accuracy'])
model.fit_generator(train_generator,epochs=5,validation_data=validation_generator)