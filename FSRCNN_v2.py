#!/usr/bin/env python

from __future__ import print_function
import tensorflow as tf
from tensorflow.image import psnr
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Input, Conv2DTranspose, PReLU
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.metrics import MeanMetricWrapper
from keras.backend import clear_session
from numpy import ceil
import matplotlib.pyplot as plt
import glob, time, os, logging


#%% Initial code setup:
print("\033[H\033[J"); time.sleep(0.1) # clear the ipython console before starting the code.
clear_session()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Add environment variables to handle OpenMP conflicts
logging.basicConfig(level=logging.INFO)


#%% USER INPUT:
#--------------------------------------------------
train_image_paths = glob.glob("./data/train/*.png")
test_image_paths = glob.glob("./data/test/Set14/*.png")
hr_img_size = (258, 258) # CHOOSE THE TARGET SIZE FOR ITS DIVISION WITH THE SCALING FACTOR TO RETURN AN EVEN NUMBER
batch_size = 64
epochs = 500
scaling = 3
aug_factor = 2 # will add to the training data X times the amount of training data
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
    img.set_shape([None, None, 1])    # Set static shape if known (e.g., for consistent pipeline behavior)
    return img

def generate_lr(hr_image, scale=2):
    """
    Generate a low-resolution version of the given high-resolution image.
    Args:
        hr_image: A high-resolution image tensor.
        scale: the downscaling factor of the high-resolution image
    Returns:
        lr_image: low-resolution image tensor.
    """
    hr_height, hr_width = tf.shape(hr_image)[0], tf.shape(hr_image)[1] # tf.shape dynamically computes the shape
    lr_height, lr_width = hr_height // scale, hr_width // scale
    lr_image = tf.image.resize(hr_image, [lr_height, lr_width], method='bicubic')
    return lr_image

# Preprocessing function for tf.data.Dataset
def preprocess(filepath, scale, target_size=(256, 256)):
    """
    Preprocesses an image by generating its low-resolution version.
    Args:
        filepath: A TensorFlow string tensor representing the file path.
        scale: The scaling factor for low-resolution generation.
    Returns:
        A tuple (lr_image, hr_image) containing the low-resolution and high-resolution images.
    """
    # Load the HR image
    hr_image = load_image(filepath)
    
    # Ensure the HR image has size matching hr_img_size input
    # if too large, image cropped, if too small, image padded
    hr_image = crop_and_pad_images(hr_image, target_size)  # Only crop/pad HR image
    
    # Generate the LR image by downscaling
    lr_image = generate_lr(hr_image, scale)
    
    return lr_image, hr_image


#%% SCALING TRAINING IMAGES:
#---------------------------
# Training images are of various size. These functions ensure their size
# become constant thanks to cropping of padding before starting the training
def crop_and_pad_images(hr_image, target_size=(256, 256)):
    """
    Crops images larger than target_size and pads images smaller than target_size.
    Cropping removes borders equally from all sides. 
    Padding adds zero-padding to make up the difference in size.

    Args:
        hr_image: High-resolution image tensor that potenttially needs to be cropped and padded
        target_size: Tuple (height, width) specifying the target dimensions.

    Returns:
        Cropped and padded HR image tensors.
    """
    def crop_or_pad(image, target_height, target_width):
        # Get current height and width
        current_height, current_width = tf.shape(image)[0], tf.shape(image)[1]
    
        # Determine offsets for cropping (if needed)
        offset_height = tf.maximum((current_height - target_height) // 2, 0)
        offset_width = tf.maximum((current_width - target_width) // 2, 0)
    
        # Crop the image if dimensions are larger than the target
        cropped_height = tf.minimum(current_height, target_height)
        cropped_width = tf.minimum(current_width, target_width)
        image = tf.image.crop_to_bounding_box(
            image,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=cropped_height,
            target_width=cropped_width)
    
        # Pad the image if dimensions are smaller than the target
        image = tf.image.pad_to_bounding_box(
            image,
            offset_height=0,
            offset_width=0,
            target_height=target_height,
            target_width=target_width)
    
        return image

    # Apply cropping and padding to both LR and HR images
    hr_image = crop_or_pad(hr_image, target_size[0], target_size[1])

    return hr_image

#%% DATA AUGMENTATION
#------------------
def augment_image(lr_image, hr_image):
    """
    Consistently augments the LR and HR images using ImageDataGenerator.
    Args:
        lr_image: Low-resolution image (NumPy array).
        hr_image: High-resolution image (NumPy array).
    Returns:
        Augmented LR and HR images.
    """
    transform = False
    if tf.random.uniform([]) > 0.5:                        # Random horizontal flip
        transform = True
        lr_image = tf.image.flip_left_right(lr_image)
        hr_image = tf.image.flip_left_right(hr_image)
    # if tf.random.uniform([]) > 0.5:                        # Random vertical flip
    #     lr_image = tf.image.flip_up_down(lr_image)
    #     hr_image = tf.image.flip_up_down(hr_image)
    if tf.random.uniform([]) > 0.5:                        # Random rotation
        transform = True
        rotations = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        lr_image = tf.image.rot90(lr_image, rotations)
        hr_image = tf.image.rot90(hr_image, rotations)
    if tf.random.uniform([]) > 0.5:                        # Random cropping (80%-100% of original size)
        transform = True
        crop_size = tf.random.uniform([], minval=0.8, maxval=1.0)
        lr_crop = tf.image.central_crop(lr_image, crop_size)
        hr_crop = tf.image.central_crop(hr_image, crop_size)
        lr_image = tf.image.resize(lr_crop, tf.shape(lr_image)[:2])
        hr_image = tf.image.resize(hr_crop, tf.shape(hr_image)[:2])
    if transform == False: # If no transformation randomly selected, force horizontal flip
        lr_image = tf.image.flip_left_right(lr_image)
        hr_image = tf.image.flip_left_right(hr_image)

    return lr_image, hr_image


def augment_dataset(dataset):
    """
    Applies augmentation to LR-HR pairs in a tf.data.Dataset.
    Args:
        dataset: A tf.data.Dataset containing (lr_image, hr_image) pairs.
    Returns:
        A tf.data.Dataset with augmented LR-HR image pairs.
    """
    new_dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    return new_dataset


def create_dataset(image_paths, scale, batch_size, target_size=(256, 256), augment=False, augment_factor=1, buffer_size=1000):
    """
    Creates a tf.data.Dataset with optional augmentation, mixing original and augmented images.
    Args:
        image_paths: List of paths to input images.
        scale: Downscaling factor for LR generation.
        batch_size: Number of images per batch.
        target_size: Tuple specifying the target size for images (height, width).
        augment: Whether to augment the dataset.
        augment_factor: How many times to augment the dataset.
        buffer_size: Size of the shuffle buffer.

    Returns:
        A tf.data.Dataset combining original and augmented images.
    """

    # Map the preprocess function --> downscale low resolution images
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda path: preprocess(path, scale, target_size), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Augment and combine datasets
    if augment:
        augmented_dataset = augment_dataset(dataset)  # Apply augmentation
        for _ in range(augment_factor - 1):  # Repeat augmentation as needed
            augmented_dataset = augmented_dataset.concatenate(augment_dataset(dataset))
        dataset = dataset.concatenate(augmented_dataset)  # Combine original and augmented
    # Dynamically augment and repeat
    # if augment:
    #     dataset = dataset.flat_map(lambda lr, hr: tf.data.Dataset.from_tensors((lr, hr)).repeat(augment_factor))
    #     dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle, batch, and prefetch
    if buffer_size > 0:  # Shuffle only if buffer_size > 0
        dataset = dataset.shuffle(buffer_size)
    # dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return dataset

#%% Metric function
def psnr_metric(y_true, y_pred):
    return psnr(y_true, y_pred, max_val=1.0)


#%% DIAGNOSTICS:
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

#%% Create train and test datasets
train_dataset = create_dataset(train_image_paths, scaling, batch_size, target_size=hr_img_size, augment=True, augment_factor=aug_factor)
test_dataset = create_dataset(test_image_paths, scaling, batch_size, target_size=hr_img_size, augment=False)

# Check the first few elements of the dataset
for lr, hr in train_dataset.take(1):
    logging.info(f"Low-res shape: {lr.shape}, High-res shape: {hr.shape}")
    
# Calculate the total number of image pairs (depends on amount of data augmentation)
flat_dataset = train_dataset.unbatch()
total_image_pairs = flat_dataset.reduce(0, lambda x, _: x + 1).numpy()
logging.info(f"Total image pairs: {total_image_pairs}")


#%% Build the model:
#-------------------
# input_img = Input(shape=(None, None, 1))
lr_img_size = tuple(dim // scaling for dim in hr_img_size) + (1,)
input_img = Input(shape=lr_img_size)

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
model = Conv2DTranspose(1, (9, 9), strides=(scaling, scaling), padding='same')(model)


#%% COMPILE THE MODEL:
output_img = model
model = Model(input_img, output_img, name="FSRCNN")
# model.load_weights('./checkpoints/weights-improvement-20-26.93.hdf5')

# Calculates PSNR between 2 tensors.The max_val argument specifies the dynamic
# range of the images. For normalized images, set max_val=1.0.
# Lambda Wrapper: this function is used to wrap psnr because Keras requires
# metrics to have two arguments (y_true, y_pred). The wrapper explicitly passes max_val
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='mse',
    metrics=[MeanMetricWrapper(psnr_metric)])

model.summary() # print the model summary


#%% MANAGE THE CHECKPOINTS:
# Save the best model based on validation loss
filepath = "./checkpoints/best_model.h5"
best_checkpoint = ModelCheckpoint(
    # filepath, monitor='val_psnr', save_best_only=True, mode='max', verbose=1) # Mode set to 'max' because PSNR improves as the model gets better
    filepath="./checkpoints/best_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Save periodic checkpoints every epoch:
periodic_checkpoint = ModelCheckpoint(filepath="./checkpoints/weights_epoch-{epoch:02d}.h5", save_freq='epoch', verbose=1)

callbacks_list = [best_checkpoint, periodic_checkpoint, tensorboard] # Define checkpoints callback
# callbacks_list = [checkpoint, plot_losses, tensorboard]


#%% Train the model
#----------------
model.fit(train_dataset,
          steps_per_epoch = ceil(total_image_pairs / batch_size),
          validation_data=test_dataset,
          epochs=epochs,
          callbacks=callbacks_list)

print("Done training!!!")

print("Saving the final model ...")
model.save('fsrcnn_model.h5')  # creates a HDF5 file 
del model  # deletes the existing model




