# Importing all necessary libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import load_img

import numpy as np
import os
from keras.models import load_model

img_width, img_height = 224, 224

train_data_dir = 'LanternflyEggImages/Training'
validation_data_dir = 'LanternflyEggImages/Testing'
nb_train_samples = 158
nb_validation_samples = 40
epochs = 100
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save('model_saved.h5')

model = load_model('model_saved.h5')

positives_dir = 'LanternflyEggImages/Testing/Positives/'
positives = os.listdir(positives_dir)
negatives_dir = 'LanternflyEggImages/Testing/Negatives/'
negatives = os.listdir(negatives_dir)

true_labels = []
predicted_labels = []

print("Running model on positives:")
for file in positives:
    true_labels.append('Positive')
    print('Loading ', file)
    image = load_img(positives_dir+file, target_size=(224, 224))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)
    label = model.predict(img)
    print("Predicted Class (0 - Negatives, 1 - Positives): ", file, " ", label[0][0], ", ", label)
    if label[0][0] >= 0.5:
        predicted_labels.append('Positive')
    else:
        predicted_labels.append('Negative')

print("Running model on negatives:")
for file in negatives:
    true_labels.append('Negative')
    image = load_img(negatives_dir+file, target_size=(224, 224))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)
    label = model.predict(img)
    print("Predicted Class (0 - Negatives, 1 - Positives): ", file, " ", label[0][0], ", ", label)
    if label[0][0] >= 0.5:
        predicted_labels.append('Positive')
    else:
        predicted_labels.append('Negative')

print(true_labels)
print(predicted_labels)

tp=0
tn=0
fp=0
fn=0
for i in range(len(true_labels)):
    tl = true_labels[i]
    pl = predicted_labels[i]
    if tl == pl:
        if tl == 'Positive':
            tp += 1
        else:
            tn += 1
    else:
        if tl == 'Positive':
            fn += 1
        else:
            fp += 1

print("True Positive = ", tp)
print("True Negative = ", tn)
print("False Positive = ", fp)
print("False Negative = ", fn)
