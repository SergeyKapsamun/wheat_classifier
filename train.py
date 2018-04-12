import tensorflow as tf
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator


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


ngpus = 1

# Initialising the CNN
if ngpus > 1:
    with tf.device('/cpu:0'):
        classifier = init_model()
else:
    classifier = init_model()

# Compiling the CNN
if ngpus > 1:
    classifier = multi_gpu_model(classifier, ngpus)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the CNN to the images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=90,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(128, 128),
    batch_size=10,
    class_mode='binary'
)

# checkpoint
filepath = "best--{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='val_acc',
    verbose=1,
    save_best_only=False,
    mode='max'
)
callbacks_list = [checkpoint]

# It's a trainirovka
classifier.fit_generator(
    training_set,
    steps_per_epoch=50,
    epochs=100,
    validation_data=test_set,
    validation_steps=30,
    callbacks=callbacks_list,
    verbose=1
)
