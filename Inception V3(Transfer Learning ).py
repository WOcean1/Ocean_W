#代码示例：6-21
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

IMSIZE=224

validation_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input).flow_from_directory(
    'E:/why/CatDog/validation',
    target_size=(IMSIZE, IMSIZE),
    batch_size=100,
    class_mode='categorical')

#代码示例：6-22
train_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.5,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True).flow_from_directory(
    'E:/why/CatDog/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size=150,
    class_mode='categorical')

#代码示例：6-23
from matplotlib import pyplot as plt
plt.ion()
plt.figure()
fig,ax = plt.subplots(2,5)
fig.set_figheight(6)
fig.set_figwidth(15)
ax=ax.flatten()
X,Y=next(train_generator)
for i in range(10): ax[i].imshow(X[i,:,:,0])
plt.savefig('E:/why/plt save/test6-4.jpg')

#代码示例：6-24
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Activation
from keras import Model

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2,activation='softmax')(x)
model=Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.summary()

#代码示例：6-25
from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.fit_generator(train_generator,epochs=5,validation_data=validation_generator)