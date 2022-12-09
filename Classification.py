import numpy
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

"""
Reference: https://www.kaggle.com/code/medianosandie/skin-diseases-classification
"""
BATCH_SIZE = 100
IMAGE_SIZE = 300
train_path = "E:\\ZZAX\\Dataset\\train"
test_path = "E:\\ZZAX\\Dataset\\test"


def train_val_generators(TRAINING_DIR, VALIDATION_DIR, IMAGE_SIZE, BATCH_SIZE):
    train_datagen = ImageDataGenerator(rescale=(1. / 255),
                                       shear_range=0.2,
                                       zoom_range=0.3,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       brightness_range=[0.2, 1.2],
                                       rotation_range=0.2,
                                       horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='categorical',
                                                        target_size=(IMAGE_SIZE, IMAGE_SIZE))

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                      batch_size=BATCH_SIZE,
                                                      class_mode='categorical',
                                                      target_size=(IMAGE_SIZE, IMAGE_SIZE))

    return train_generator, test_generator


train_generator, test_generator = train_val_generators(train_path, test_path,IMAGE_SIZE,BATCH_SIZE)
train_img, train_label = train_generator.next()
test_img, test_label = test_generator.next()
print(type(train_img), train_img.shape, type(train_label), train_label.shape)
# print(train_label[0])
# plt.imshow(train_img[0])
# plt.show()

class_names = train_generator.class_indices
NUMBER_OF_CLASSES = len(class_names)
print(f'number of classes : {NUMBER_OF_CLASSES}')

#--Segment
import app
print(train_img)
for i in range(len(train_img)):
    img = train_img[i] * 255
    print(img)
    train_img[i] = app.ycrcb_color_space_segmentation(img) * 1./255
    print(train_img[i])

print(train_img)

#--Start
y_train = np.zeros(len(train_label))
y_test = np.zeros(len(test_label))
for i in range(len(train_label)):
    y_train[i] = np.sum(train_label[i])
for i in range(len(test_label)):
    y_test[i] = np.sum(test_label[i])

x_train = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))
for i in range(len(train_img)):
    img = train_img[i]
    grayscale_img = np.average(img, axis=-1)
    x_train[i, :, :] = grayscale_img
x_test = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))
for i in range(len(test_img)):
    img = test_img[i]
    grayscale_img = np.average(img, axis=-1)
    x_test[i, :, :] = grayscale_img

#--normalize
x_test = x_test/x_test.max()
x_train = x_train/x_train.max()

#--reshape
# print('x_train shape= ', x_train.shape)
x_train = x_train.reshape(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,1)
# print('x_train re-shape= ', x_train.shape)

# print('x_test shape= ', x_test.shape)
x_test = x_test.reshape(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,1)
# print('x_test re-shape= ', x_test.shape)

#--convert y_train and y_test to be one-hot encoded for categorical analysis
from keras.utils.np_utils import to_categorical
y_cat_test = to_categorical(y_test,10)
# print('y_cat_test shape= ', y_cat_test.shape)
y_cat_train = to_categorical(y_train,10)
# print('y_train[0]', y_train[0])
# print('y_cat_train[0]', y_cat_train[0])

#=================================
#--- build the model
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten

model = Sequential()
# Convolution layer
model.add(Conv2D(filters=BATCH_SIZE, kernel_size=(3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), activation='relu'))
# Pooling layer
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
# dense layer
model.add(Dense(128, activation='relu'))
# create the output layer
model.add(Dense(10, activation='softmax'))
# compile the model
model.compile(loss='categorical_crossentropy', optimizer = 'rmsprop', metrics =['accuracy'])
model.summary()

#----visualize the model
import visualkeras
from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 32)
visualkeras.layered_view(model, legend=True, draw_volume=True, font=font, to_file='DL_layer_volume.png').show()
visualkeras.layered_view(model, legend=True, draw_volume=False, font=font, to_file='DL_layer.png').show()

#========================
#----train and evaluate the model

#----Train the model
model.fit(x_train,y_cat_train,verbose=1,epochs=2) # train the model
evalResults= model.evaluate(x_test,y_cat_test) # evaluate the model
print(model.metrics_names,'=', evalResults)

#---use the model on images not seen before
from sklearn.metrics import classification_report
predictions = model.predict(x_test) # results in y_test
predicted_classes=np.argmax(predictions,axis=1)
print(classification_report(y_test,predicted_classes))
