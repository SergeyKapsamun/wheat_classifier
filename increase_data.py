from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=270,
    horizontal_flip=True)

i = 0
for training_set in train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(182, 182),
        batch_size=1000,
        class_mode="binary",
        save_to_dir='dataset/training_set/augmented'):
    i += 1
    if i == 1:
        break
