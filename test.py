import os
import numpy as np
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers import MaxPooling2D
from keras.layers import Conv2D, Dropout


# ALL PARALLEL!!!!111111[]1[1][1]1
def init_model():
    # Initialising the CNN
    model = Sequential()

    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))

    # Step 2 - Pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Adding a second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Step 3 - Flattening
    model.add(Flatten())

    # Step 4 - Full connection
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    return model


# Making new predictions
classifier = init_model()
classifier.load_weights('best--1.00.hdf5')

for d, dirs, files in os.walk('dataset/demo'):
    for f in files:
        test_image = image.load_img("".join(['dataset/demo/', f]), target_size=(128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict(test_image)

        # training_set.class_indices
        if result[0][0] == 1:
            prediction = 'good'
        else:
            prediction = 'bad'
        print(f, prediction, result[0][0], sep="->")

