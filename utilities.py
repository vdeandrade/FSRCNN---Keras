# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:21:27 2024

@author: Vincent
"""

from PIL import Image
import os

#%% Convert bmp to png
file_path = r"D:\vdeandrade\Deep_learning\GitHub_repos\FSRCNN---Keras\data\test\Set14"
file_path = r"D:\vdeandrade\Deep_learning\GitHub_repos\FSRCNN---Keras\data\test\Set5"
file_list = os.listdir(file_path)
for i in file_list:
    source_files = os.path.join(file_path, i)
    img = Image.open(source_files)
    file_name = i[:-4]+'.png'
    file_name_saved = os.path.join(file_path, file_name)
    img.save(file_name_saved)
    # print(file_name_saved)

#%%
cnt=0
for i in range(3):
    cnt+=1
    print(cnt)

#%% Displaying Basic Information:
# You can loop through the dataset and print out details about each element,
# such as shape and dtype:
for element in train_dataset.take(5):  # Limit the output to the first 5 elements
    lr_image, hr_image = element
    print("Low-Resolution Image Shape:", lr_image.shape)
    print("Low-Resolution Image Type:", lr_image.dtype)
    print("High-Resolution Image Shape:", hr_image.shape)
    print("High-Resolution Image Type:", hr_image.dtype)
    print("------")

#%%
for element in test_dataset.take(5):  # Limit the output to the first 5 elements
    lr_image, hr_image = element
    print("Low-Resolution Image Shape:", lr_image.shape)
    print("Low-Resolution Image Type:", lr_image.dtype)
    print("High-Resolution Image Shape:", hr_image.shape)
    print("High-Resolution Image Type:", hr_image.dtype)
    print("------")


#%%
# Exploring Dataset Elements:
#----------------------------
# Print Element Types and Shapes: You can print the types and shapes of the elements
# in the dataset. This will give you an idea of what kind of data is stored
# in the dataset and how it's structured.

def inspect_dataset(dataset, num_elements=5):
    for element in dataset.take(num_elements):
        if isinstance(element, (tuple, list)):
            for i, el in enumerate(element):
                print(f"Element {i + 1}:")
                print(f"  Type: {type(el)}")
                print(f"  Shape: {el.shape}")
                print(f"  DType: {el.dtype}")
                print("------")
        else:
            print("Single Element:")
            print(f"  Type: {type(element)}")
            print(f"  Shape: {element.shape}")
            print(f"  DType: {element.dtype}")
            print("------")

# Call the function to inspect the dataset
inspect_dataset(dataset)


# Convert Tensor to Numpy for Further Inspection: To get more detailed insights,
# you can convert TensorFlow tensors to NumPy arrays and inspect their content.
def inspect_dataset_content(dataset, num_elements=5):
    for element in dataset.take(num_elements):
        if isinstance(element, (tuple, list)):
            for i, el in enumerate(element):
                el_np = el.numpy()
                print(f"Element {i + 1} content (first few values): {el_np.flat[:5]}")
        else:
            element_np = element.numpy()
            print(f"Single Element content (first few values): {element_np.flat[:5]}")

# Call the function to inspect the content of the dataset
inspect_dataset_content(dataset)

# Summarize Dataset: Create a summary function to provide a comprehensive overview
# of the dataset, including the shapes and types of the first few elements.
def summarize_dataset(dataset, num_elements=5):
    print("Dataset Summary:")
    for element in dataset.take(num_elements):
        if isinstance(element, (tuple, list)):
            for i, el in enumerate(element):
                print(f"Element {i + 1}:")
                print(f"  Type: {type(el)}")
                print(f"  Shape: {el.shape}")
                print(f"  DType: {el.dtype}")
                print(f"  Content (first few values): {el.numpy().flat[:5]}")
                print("------")
        else:
            print("Single Element:")
            print(f"  Type: {type(element)}")
            print(f"  Shape: {element.shape}")
            print(f"  DType: {element.dtype}")
            print(f"  Content (first few values): {element.numpy().flat[:5]}")
            print("------")

# Call the function to summarize the dataset
summarize_dataset(dataset)

#%%
input_size = 85
strides = 3
kernel_size = 12
padding = (kernel_size - 1) // 2
output_size = (input_size - 1) * strides + kernel_size - 2 * padding
print(output_size)


