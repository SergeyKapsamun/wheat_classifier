import tensorflow as tf
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


def init_model():
    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape=(182, 182, 3), activation='relu'))
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))
    # Adding a second convolutional layer
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=1, activation='sigmoid'))
    return classifier

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
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# checkpoint
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
rotation_range=90,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
target_size = (182, 182),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
target_size = (182, 182),
batch_size = 10,
class_mode = 'binary')
filepath = "best--{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
classifier.fit_generator(training_set,
steps_per_epoch = 50,
epochs = 100,
validation_data = test_set,
validation_steps = 30, callbacks=callbacks_list, verbose=1)