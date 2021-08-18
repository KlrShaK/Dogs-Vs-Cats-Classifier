# todo Use Full dataset and add augmentation to the dataset

import tensorflow as tf
import numpy as np


class myCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') >= 0.70:
            print('Training Stopped as accuracy reached >72%!!')
            self.model.stop_training = True
callbacks = myCallbacks()

# Path of testing and training Directories
train_dir = r'dogs_vs_cats_dataset_full\train'
test_dir = r'dogs_vs_cats_dataset_full\validation'

model = tf.keras.Sequential(
    [
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.BatchNormalization(trainable=True),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(trainable=True),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)
model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer= RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Making Training Image Generator and Validation Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size= 20,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size= (150,150),
    batch_size= 20,
    class_mode= 'binary'
)


history = model.fit(train_generator, epochs=100, steps_per_epoch=100, validation_data=validation_generator, validation_steps=50, verbose=1, callbacks=[callbacks])

from acc_plotter import plot_accuracy
plot_accuracy(history)

from tensorflow.keras.preprocessing import image

choice = input('Do you want to Test on a Images??(y/n): ')
choice = choice.lower()
while choice == 'y':
    try:
        path = input('Input the Path to the image  file :')
        img = image.load_img(path, target_size=(150, 150))
    except:
        print("error in user input!")
        path = input('Input the Path to the image  file :')
        img = image.load_img(path, target_size=(150, 150))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    result = model.predict(images)
    print(result[0])
    if result[0] > 0.5:
        print('Image is Dog')
    elif result[0] < 0.5:
        print('Image is Cat')
    choice = input('Do you want to Test more Images??(y/n): ')
    choice = choice.lower()
