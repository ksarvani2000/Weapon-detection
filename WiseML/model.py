# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



base_dir = os.path.join(os.getcwd(), 'weapon-detection')
train_dir = os.path.join(base_dir, 'train')
knifes = os.path.join(train_dir, 'knifes')
guns = os.path.join(train_dir, 'guns')
total = len(os.listdir(knifes)) + len(os.listdir(guns))
print(total)
test_dir = os.path.join(base_dir, 'test')
knifes1 = os.path.join(test_dir, 'knifes')
guns1 = os.path.join(test_dir, 'guns')
total1 = len(os.listdir(knifes1)) + len(os.listdir(guns1))
print(total1)

x = []
y = []
m,n = 100,100
count = 0
classes = os.listdir(train_dir)
for i in classes:
  print(i)
  imgfiles = os.listdir(train_dir + '/' +i);
  for j in imgfiles:
    try:
      img = Image.open(train_dir+'/'+ i +'/'+ j, mode = 'r');
      img = img.convert(mode='RGB')
      img = img.resize((m,n))
      img = np.array(img)
      x.append(img)
      y.append(count)
    except:
      print("Error")
  count += 1

x = np.array(x)
y = np.array(y)
print(x)
print(y)

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)

BATCH_SIZE = 120
IMG_SHAPE = x_train.shape[1:]
print(IMG_SHAPE)
EPOCH = 30

classifier = Sequential()

classifier.add(Conv2D(20, (3, 3), input_shape= x_train.shape[1:]))
classifier.add(Activation('relu'))

classifier.add(MaxPooling2D(2, 2))

classifier.add(Conv2D(20, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(2, 2))

classifier.add(Conv2D(20, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(2, 2))

classifier.add(Conv2D(20, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(2, 2))

classifier.add(Conv2D(20, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(2, 2))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=2, activation='softmax'))

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

classifier.summary()

history = classifier.fit(x_train,y_train,batch_size=32,epochs=6,verbose=1,validation_data=(x_test,y_test))

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(6)

plt.figure(figsize=(8,8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title("Training vs Validation Accuracy of CNN")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='lower right')
plt.title("Training vs Validation Loss of CNN")

plt.show()
plt.close()

classifier.save('weapon_detection_model.h5')