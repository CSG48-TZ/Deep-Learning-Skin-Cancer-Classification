import numpy as np
import matplotlib.pyplot as plt
import cv2

derm_eye = cv2.imread('E:\\ZZAX\\Dataset\\test\\Acne and Rosacea Photos\\07PerioralDermEye.jpg')
derm_eye = cv2.cvtColor(derm_eye,cv2.COLOR_BGR2RGB)
plt.imshow(derm_eye)
plt.show()
perioral = cv2.imread('E:\\ZZAX\\Dataset\\test\\Acne and Rosacea Photos\\07SteroidPerioral.jpg')
perioral = cv2.cvtColor(perioral,cv2.COLOR_BGR2RGB)
plt.imshow(perioral)
plt.show()

#--generate different images from each image
from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range =30, width_shift_range =0.1,
                               height_shift_range =0.1, rescale =1/255,shear_range =0.2,
                               zoom_range =0.2,horizontal_flip=True,
                               fill_mode ='nearest')
for i in range(5):
    randImg =image_gen.random_transform(perioral)
    plt.imshow(randImg)
    plt.show()
#-----------
numOfImages= image_gen.flow_from_directory('E:\\ZZAX\\Dataset\\test')
print(numOfImages)
"""
#-----build the model
from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Conv2D,MaxPooling2D,Dense

model =Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3), input_shape=(150,150,3), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#--add two more layers----
model.add(Conv2D(filters=64,kernel_size=(3,3), input_shape=(150,150,3), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3), input_shape=(150,150,3), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))


model.add(Dropout(0.5)) #-- prevent over fitting by randomly turning off 50% of the neurons

model.add(Dense(1, activation='sigmoid')) # last layer 0,1 either cat or dog

#--compile the model
model.compile(loss='binary_crossentropy',optimizer ='adam',metrics=['accuracy'])
print(model.summary())

#--- Train the model

batch_size =16
train_image_gen = image_gen.flow_from_directory('E:\\ZZAX\\Dataset\\train', target_size=[150,150], batch_size=batch_size,class_mode='binary')
print(train_image_gen.class_indices)

test_image_gen = image_gen.flow_from_directory('E:\\ZZAX\\Dataset\\test', target_size=[150,150], batch_size=batch_size,class_mode='binary')

print(test_image_gen.classes)

# during an iteration take 15 batches, here batch size is 16
results = model.fit_generator(train_image_gen, epochs=3,steps_per_epoch=15, validation_data=test_image_gen,validation_steps=12)
print(results.history)
plt.plot(results.history['accuracy'])
plt.show()
#----load already trained model-------------
from keras.models import load_model
new_model = load_model('C:/Users\Hassan Hajjdiab/Desktop/My Udemy courses/Python-OpenCV-DeepLearning/06-Deep-Learning-Computer-Vision/cat_dog_100epochs.h5')

#--select a dog image
dog_file= 'C:/Users/Hassan Hajjdiab/Desktop/My Udemy courses/CV2Project/CATS_DOGS/test/DOG/10005.jpg'
from keras.preprocessing import image
dog_img = image.image_utils.load_img(dog_file,target_size=(150,150)) # resize image to be 150 X 150
dog_img = image.image_utils.img_to_array(dog_img) # save image as array
print(dog_img.shape) # should be (150,150,3)
dog_img = np.expand_dims(dog_img,axis=0) # change to batch of 1 image, shape is (1,150,150,3)
print(dog_img.shape)

dog_img = dog_img/255  # normalize image to values between 0 to 255
print(new_model.predict(dog_img)) # mew model predictions

print(model.predict(dog_img)) # old model predictions
"""