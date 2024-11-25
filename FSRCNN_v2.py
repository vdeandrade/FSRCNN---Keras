#!/usr/bin/env python

from __future__ import print_function
import tensorflow as tf
from tensorflow.image import psnr
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Input, Conv2DTranspose, PReLU
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import glob

#%% USER INPUT:
#--------------------------------------------------
train_image_paths = glob.glob("./data/train/*.png")
test_image_paths = glob.glob("./data/test/Set14/*.bmp")
batch_size = 64
epochs = 500
im_scaling = 3
aug_factor = 5 # will add to the training data X the amount of traning data
#--------------------------------------------------

#%% FUNCTIONS:
#-------------
# Load an image and preprocess it
def load_image(filepath):
    """
    Loads an image, decodes it to grayscale, and normalizes pixel values to [0, 1].
    """
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=1)  # Convert to grayscale
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize to [0, 1]
    return img

def generate_lr(hr_image, scale=2):
    """
    Generate a low-resolution version of the given high-resolution image.
    Args:
        hr_image: Tensor, high-resolution image.
        scale: Int, the downscaling factor.
    Returns:
        lr_image: Tensor, low-resolution image.
    """
    hr_height, hr_width = tf.shape(hr_image)[0], tf.shape(hr_image)[1]
    lr_height, lr_width = hr_height // scale, hr_width // scale
    lr_image = tf.image.resize(hr_image, [lr_height, lr_width], method='bicubic')
    return lr_image

# Preprocessing function for tf.data.Dataset
def preprocess(filepath, scale):
    """
    Takes a file path, loads the HR image, and generates an LR image.
    """
    hr_image = load_image(filepath)  # Load HR image
    lr_image = generate_lr(hr_image, scale)  # Generate LR image
    return lr_image, hr_image


# DATA AUGMENTATION
#------------------
datagen = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

def augment_image(lr_image, hr_image, datagen):
    """
    Consistently augments the LR and HR images using ImageDataGenerator.
    Args:
        lr_image: Low-resolution image (NumPy array).
        hr_image: High-resolution image (NumPy array).
        datagen: An instance of ImageDataGenerator for applying augmentations.
    Returns:
        Augmented LR and HR images.
    """
    # Convert tensors to NumPy arrays for compatibility with ImageDataGenerator
    lr_image = lr_image.numpy()
    hr_image = hr_image.numpy()

    # Generate consistent transformation parameters
    params = datagen.get_random_transform(lr_image.shape)

    # Apply the same transformation to both images
    lr_aug = datagen.apply_transform(lr_image, params)
    hr_aug = datagen.apply_transform(hr_image, params)

    return lr_aug, hr_aug

def augment_dataset(dataset, datagen):
    """
    Applies augmentation to a tf.data.Dataset of LR-HR pairs.
    Args:
        dataset: A tf.data.Dataset containing (lr_image, hr_image) pairs.
        datagen: An ImageDataGenerator instance for applying augmentations.
    Returns:
        A tf.data.Dataset with augmented LR-HR pairs.
    """
    # tf.py_function wraps the augment_image function so it can operate within TensorFlow’s tf.data.Dataset pipeline
    augmented_dataset = dataset.map(lambda lr, hr: tf.py_function(
                func=augment_image, 
                inp=[lr, hr, datagen], 
                Tout=[tf.float32, tf.float32]),
                num_parallel_calls=tf.data.AUTOTUNE)
    
    return augmented_dataset

def create_dataset(image_paths, scale, batch_size, augment=False, augment_factor=aug_factor, buffer_size=tf.data.AUTOTUNE):
    """
    Creates a tf.data.Dataset with optional augmentation, mixing original and augmented images.
    """
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda path: preprocess(path, scale), num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        augmented_dataset = augment_dataset(dataset)  # Apply dynamic augmentation
        for _ in range(augment_factor - 1):  # Repeat augmentation as needed
            dataset = dataset.concatenate(augment_dataset(augmented_dataset))
    
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

#%% DIAGNOSTICS:

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

import matplotlib.pyplot as plt
from keras.callbacks import Callback

class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.psnr = []
        self.val_psnr = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.psnr.append(logs.get('psnr'))
        self.val_psnr.append(logs.get('val_psnr'))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.legend()
        plt.title('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(self.psnr, label='Training PSNR')
        plt.plot(self.val_psnr, label='Validation PSNR')
        plt.legend()
        plt.title('PSNR')

        plt.show()

plot_losses = PlotLosses()

#%% Create train and test datasets
train_dataset = create_dataset(train_image_paths, scale=im_scaling, batch_size=batch_size, augment=True, augment_factor=aug_factor)
test_dataset = create_dataset(test_image_paths, scale=im_scaling, batch_size=batch_size, augment=False, augment_factor=aug_factor)

disp_batch_num = 0 # index of the batch choosen to display an example of augmented low and high res images
for lr, hr in train_dataset.take(disp_batch_num): # squeeze removes the batch and channel dimensions
    plt.subplot(1,2,1), plt.imshow(lr[0].numpy().squeeze(), cmap='gray'), plt.title("Low res trained image of batch #%i" % disp_batch_num)
    plt.subplot(1,2,2), plt.imshow(hr[0].numpy().squeeze(), cmap='gray'), plt.title("Low res trained image of batch #%i" % disp_batch_num), plt.show()


#%% Build the model:
#-------------------
input_img = Input(shape=(None, None, 1))
# 1) Feature extraction layer:
model = Conv2D(56, (5, 5), padding='same', kernel_initializer='he_normal')(input_img)
model = PReLU()(model)
# 2) Shrinking layer reducing feature dimensions:
model = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(model)
model = PReLU()(model)
# 3) Multiple mapping layers perform transformations in the low-dimensional space:
model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = PReLU()(model)
model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = PReLU()(model)
model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = PReLU()(model)
model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = PReLU()(model)
# 4) Expanding layer restoring the dimensionality:
model = Conv2D(56, (1, 1), padding='same', kernel_initializer='he_normal')(model)
model = PReLU()(model)
# 5) Deconvolution layer performs upscaling to obtain the final HR image
model = Conv2DTranspose(1, (9, 9), strides=(4, 4), padding='same')(model)


#%%
output_img = model
model = Model(input_img, output_img)
# model.load_weights('./checkpoints/weights-improvement-20-26.93.hdf5')

# Calculates PSNR between 2 tensors.The max_val argument specifies the dynamic
# range of the images. For normalized images, set max_val=1.0.
# Lambda Wrapper: this function is used to wrap psnr because Keras requires
# metrics to have two arguments (y_true, y_pred). The wrapper explicitly passes max_val

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='mse',
    metrics=[lambda y_true, y_pred: psnr(y_true, y_pred, max_val=1.0)])

model.summary() # print the model summary

#%% Define checkpoint callback
filepath = "./checkpoints/weights-improvement-{epoch:02d}-{val_psnr:.2f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_psnr', save_best_only=True, mode='max', verbose=1)
callbacks_list = [checkpoint, plot_losses, tensorboard]


#%%
filepath = "./checkpoints/best_model.h5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_psnr', save_best_only=True, mode='max', verbose=1) # Mode set to 'max' because PSNR improves as the model gets better
    # filepath, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

callbacks_list = [checkpoint]

# Train the model
#----------------
model.fit(train_dataset,
          steps_per_epoch = len(train_image_paths) // batch_size,
          validation_data=test_dataset,
          epochs=epochs,
          callbacks=callbacks_list)

print("Done training!!!")

print("Saving the final model ...")
model.save('fsrcnn_model.h5')  # creates a HDF5 file 
del model  # deletes the existing model




