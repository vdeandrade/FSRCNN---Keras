#!/usr/bin/env python

from __future__ import print_function
import tensorflow as tf
from tensorflow.image import psnr
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Input, Conv2DTranspose, PReLU
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
import matplotlib.pyplot as plt
import glob, time


#%%
print("\033[H\033[J"); time.sleep(0.1) # clear the ipython console before starting the code.

#%% USER INPUT:
#--------------------------------------------------
train_image_paths = glob.glob("./data/train/*.png")
test_image_paths = glob.glob("./data/test/Set14/*.bmp")
img_size = (256, 256, 1)
batch_size = 64
epochs = 500
im_scaling = 3
aug_factor = 5 # the size of the training dataset will be increase "aug_factor" times
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
def preprocess(filepath, scale):
    """
    Preprocesses an image by generating its low-resolution version.
    Args:
        filepath: A TensorFlow string tensor representing the file path.
        scale: The scaling factor for low-resolution generation.
    Returns:
        A tuple (lr_image, hr_image) containing the low-resolution and high-resolution images.
    """
    hr_image = load_image(filepath)  # Load HR image
    lr_image = generate_lr(hr_image, scale)  # Generate LR image
    return lr_image, hr_image


#%% SCALING TRAINMING IMAGES:
#----------------------------
# Training images are of various size. These functions ensure their size
# become constant thanks to cropping of padding before starting the training
def crop_and_pad_images(lr_image, hr_image, target_size=(256, 256)):
    """
    Crops images larger than target_size and pads images smaller than target_size.
    Cropping removes borders equally from all sides.
    Padding adds zero-padding to make up the difference in size.

    Args:
        lr_image: Low-resolution image tensor.
        hr_image: High-resolution image tensor.
        target_size: Tuple (height, width) specifying the target dimensions.

    Returns:
        Cropped and padded LR and HR image tensors.
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
    lr_image = crop_or_pad(lr_image, target_size[0], target_size[1])
    hr_image = crop_or_pad(hr_image, target_size[0], target_size[1])

    return lr_image, hr_image

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
    if tf.random.uniform([]) > 0.5:                        # Random horizontal flip
        lr_image = tf.image.flip_left_right(lr_image)
        hr_image = tf.image.flip_left_right(hr_image)
    # if tf.random.uniform([]) > 0.5:                        # Random vertical flip
    #     lr_image = tf.image.flip_up_down(lr_image)
    #     hr_image = tf.image.flip_up_down(hr_image)
    if tf.random.uniform([]) > 0.5:                        # Random rotation
        rotations = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        lr_image = tf.image.rot90(lr_image, rotations)
        hr_image = tf.image.rot90(hr_image, rotations)
    if tf.random.uniform([]) > 0.5:                        # Random cropping (80%-100% of original size)
        crop_size = tf.random.uniform([], minval=0.8, maxval=1.0)
        lr_crop = tf.image.central_crop(lr_image, crop_size)
        hr_crop = tf.image.central_crop(hr_image, crop_size)
        lr_image = tf.image.resize(lr_crop, tf.shape(lr_image)[:2])
        hr_image = tf.image.resize(hr_crop, tf.shape(hr_image)[:2])

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


def create_dataset(image_paths, scale, batch_size, target_size=(256, 256), augment=False, augment_factor=2, buffer_size=1000):
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
    def preprocess_with_crop_and_pad(filepath, scale):
        hr_image = load_image(filepath)
        lr_image = generate_lr(hr_image, scale)
        lr_image, hr_image = crop_and_pad_images(lr_image, hr_image, target_size)
        return lr_image, hr_image

    # Load and preprocess images
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(
        lambda path: preprocess_with_crop_and_pad(path, scale),
        num_parallel_calls=tf.data.AUTOTUNE)
    
    # Augment and combine datasets
    if augment:
        augmented_dataset = augment_dataset(dataset)  # Apply augmentation
        for _ in range(augment_factor - 1):  # Repeat augmentation as needed
            augmented_dataset = augmented_dataset.concatenate(augment_dataset(dataset))
        dataset = dataset.concatenate(augmented_dataset)  # Combine original and augmented

    # Shuffle, batch, and prefetch
    if buffer_size > 0:  # Shuffle only if buffer_size > 0
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


#%% DIAGNOSTICS:
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.losses, self.val_losses, self.psnr, self.val_psnr = [], [], [], []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.psnr.append(logs.get('psnr'))
        self.val_psnr.append(logs.get('val_psnr'))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1), plt.plot(self.losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss'), plt.legend(), plt.title('Loss')

        plt.subplot(1, 2, 2), plt.plot(self.psnr, label='Training PSNR')
        plt.plot(self.val_psnr, label='Validation PSNR'), plt.legend(), plt.title('PSNR'), plt.show()

plot_losses = PlotLosses()

#%% Create train and test datasets
train_dataset = create_dataset(train_image_paths, im_scaling, batch_size, target_size=img_size, augment=True, augment_factor=aug_factor)
test_dataset = create_dataset(test_image_paths, im_scaling, batch_size, target_size=img_size, augment=False)

# Check the first few elements of the dataset
# for lr, hr in train_dataset.take(1):
#     print("Low-res shape:", lr.shape)
#     print("High-res shape:", hr.shape)


disp_batch_num = 0 # index of the batch choosen to display an example of augmented low and high res images
for lr, hr in train_dataset.take(disp_batch_num): # squeeze removes the batch and channel dimensions
    plt.subplot(1,2,1), plt.imshow(lr[0].numpy().squeeze(), cmap='gray'), plt.title("Low res trained image of batch #%i" % disp_batch_num)
    plt.subplot(1,2,2), plt.imshow(hr[0].numpy().squeeze(), cmap='gray'), plt.title("Low res trained image of batch #%i" % disp_batch_num), plt.show()


#%% Build the model:
#-------------------
# input_img = Input(shape=(None, None, 1))
input_img = Input(shape=img_size)

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




