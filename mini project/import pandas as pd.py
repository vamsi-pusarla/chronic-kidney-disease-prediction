import pandas as pd
import numpy as np
import os
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt


from numpy.random import seed
seed(101)


import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# Define base directory
base_dir = 'dataset_dir'

# Check if base directory exists, if not, create it
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
else:
    print(f"Directory '{base_dir}' already exists.")

# Define train and validation directories
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')

# Check if train directory exists, if not, create it
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
else:
    print(f"Directory '{train_dir}' already exists.")

# Check if validation directory exists, if not, create it
if not os.path.exists(val_dir):
    os.mkdir(val_dir)
else:
    print(f"Directory '{val_dir}' already exists.")

# Create subdirectories for each class in training and validation directories
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# Create directories in train_dir
for class_name in class_names:
    class_dir = os.path.join(train_dir, class_name)
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
    else:
        print(f"Directory '{class_dir}' already exists.")

# Create directories in val_dir
for class_name in class_names:
    class_dir = os.path.join(val_dir, class_name)
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
    else:
        print(f"Directory '{class_dir}' already exists.")
df_data = pd.read_csv('C:/Users/v mouli/Downloads/skind/HAM10000_metadata.csv')
df_data.head()
df = df_data.groupby('lesion_id').count()
df = df[df['image_id'] == 1]
df.reset_index(inplace=True)
df.head()
def identify_duplicates(x):
    
    unique_list = list(df['lesion_id'])
    
    if x in unique_list:
        return 'no_duplicates'
    else:
        return 'has_duplicates'

df_data['duplicates'] = df_data['lesion_id']

df_data['duplicates'] = df_data['duplicates'].apply(identify_duplicates)

df_data.head()
df_data['duplicates'].value_counts()
df = df_data[df_data['duplicates'] == 'no_duplicates']
df.shape
y = df['dx']
_, df_val = train_test_split(df, test_size=0.17, random_state=101, stratify=y)
df_val.shape
df_val['dx'].value_counts()

def identify_val_rows(x):
    # create a list of all the lesion_id's in the val set
    val_list = list(df_val['image_id'])
    
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'


df_data['train_or_val'] = df_data['image_id']
df_data['train_or_val'] = df_data['train_or_val'].apply(identify_val_rows)
df_train = df_data[df_data['train_or_val'] == 'train']


print(len(df_train))
print(len(df_val))
df_train['dx'].value_counts()
df_val['dx'].value_counts()
df_data.set_index('image_id', inplace=True)

folder_1 = os.listdir('C:/Users/v mouli/Downloads/skind/ham10000_images_part_1')
folder_2 = os.listdir('C:/Users/v mouli/Downloads/skind/ham10000_images_part_2')

train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

for image in train_list:
    
    fname = image + '.jpg'
    label = df_data.loc[image,'dx']
    
    if fname in folder_1:
       
        src = os.path.join('C:/Users/v mouli/Downloads/skind/ham10000_images_part_1', fname)
        dst = os.path.join(train_dir, label, fname)
        shutil.copyfile(src, dst)

    if fname in folder_2:
        
        src = os.path.join('C:/Users/v mouli/Downloads/skind/ham10000_images_part_2', fname)
        dst = os.path.join(train_dir, label, fname)
        shutil.copyfile(src, dst)
for image in val_list:
    
    fname = image + '.jpg'
    label = df_data.loc[image,'dx']
    
    if fname in folder_1:
       
        src = os.path.join('C:/Users/v mouli/Downloads/skind/ham10000_images_part_1', fname)
        dst = os.path.join(val_dir, label, fname)
        shutil.copyfile(src, dst)

    if fname in folder_2:
        src = os.path.join('C:/Users/v mouli/Downloads/skind/ham10000_images_part_2', fname)
        dst = os.path.join(val_dir, label, fname)
        shutil.copyfile(src, dst)
import os

base_dir = 'dataset_dir'
train_dir = os.path.join(base_dir, 'train_dir')

# Define the class names
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# Create a dictionary to store the counts for each class
class_counts_train = {}

# Loop through the class names
# Loop through the class names
import os

# Define the base directory
base_dir = 'dataset_dir'
train_dir = os.path.join(base_dir, 'train_dir')

# Define the class names
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# Loop through the class names
# Loop through the class names
for class_name in class_names:
    # Get the path to the class directory
    class_dir = os.path.join(train_dir, class_name)
    
    try:
        # Count the number of files in the class directory
        num_files = len(os.listdir(class_dir))
        print(f"Number of files in {class_name} directory: {num_files}")
        # Store the count in the dictionary
        class_counts_train[class_name] = num_files 
    except FileNotFoundError:
        print(f"Directory not found: {class_dir}")


# Print the counts for each class
for class_name, count in class_counts_train.items():
    print(f"{class_name}: {count}")
import os

base_dir = 'dataset_dir'  # Fixed a typo here from 'datase_dir' to 'dataset_dir'
val_dir = os.path.join(base_dir, 'val_dir')

# Define the class names
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# Create a dictionary to store the counts for each class
class_counts_val = {}

# Loop through the class names
# Loop through the class names
for class_name in class_names:
    # Get the path to the class directory
    class_dir = os.path.join(val_dir, class_name)
    
    try:
        # Count the number of files in the class directory
        num_files = len(os.listdir(class_dir))
        print(f"Number of files in {class_name} directory: {num_files}")
        # Store the count in the dictionary
        class_counts_val[class_name] = num_files 
    except FileNotFoundError:
        print(f"Directory not found: {class_dir}")


# Print the counts for each class
for class_name, count in class_counts_val.items():
    print(f"{class_name}: {count}")

import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
base_dir = 'dataset_dir'
class_list = ['mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
aug_dir = 'aug_dir'

# Create 'aug_dir' if it does not exist
if not os.path.exists(aug_dir):
    os.mkdir(aug_dir)

for item in class_list:
    img_dir = os.path.join(aug_dir, 'img_dir')
    
    # Ensure img_dir is fresh each time
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.mkdir(img_dir)

    img_class = item
    img_list = os.listdir(os.path.join(base_dir, 'train_dir', img_class))
    for fname in img_list:
        src = os.path.join(base_dir, 'train_dir', img_class, fname)
        dst = os.path.join(img_dir, fname)
        shutil.copyfile(src, dst)
    
    path = aug_dir
    save_path = os.path.join('dataset_dir', 'train_dir', img_class)

    # Create a data generator
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(
        path,
        save_to_dir=save_path,
        save_format='jpg',
        target_size=(224, 224),
        batch_size=batch_size)

    num_aug_images_wanted = 6000
    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted - num_files) / batch_size))

    for _ in range(num_batches):
        imgs, labels = next(aug_datagen)

# Clean up 'aug_dir' at the end
shutil.rmtree(aug_dir)
import os

base_dir = 'dataset_dir'
train_dir = os.path.join(base_dir, 'train_dir')

# Define the class names
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# Create a dictionary to store the counts for each class
class_counts_train = {}

# Loop through the class names
for class_name in class_names:
    # Get the path to the class directory
    class_dir = os.path.join(train_dir, class_name)
    # Count the number of files in the class directory
    num_files = len(os.listdir(class_dir))
    # Store the count in the dictionary
    class_counts_train[class_name] = num_files

# Print the counts for each class
for class_name, count in class_counts_train.items():
    print(f"{class_name}: {count}")
import os

base_dir = 'dataset_dir'  # Fixed a typo here from 'datase_dir' to 'dataset_dir'
val_dir = os.path.join(base_dir, 'val_dir')

# Define the class names
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# Create a dictionary to store the counts for each class
class_counts_val = {}

# Loop through the class names
for class_name in class_names:
    # Get the path to the class directory
    class_dir = os.path.join(val_dir, class_name)
    
    # Count the number of files in the class directory and store the count in the dictionary
    class_counts_val[class_name] = len(os.listdir(class_dir))

# Print the counts for each class
for class_name, count in class_counts_val.items():
    print(f"{class_name}: {count}")
# plots images with labels within jupyter notebook
# source: https://github.com/smileservices/keras_utils/blob/master/utils.py

# Assuming imgs is supposed to contain image data

train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 10
val_batch_size = 10
image_size = 224

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.applications.mobilenet as mobilenet

# Define the base directory
base_dir = 'dataset_dir'

# Define paths for training and validation directories within the base directory
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')

# Get the number of samples in training and validation sets
num_train_samples = sum(len(files) for _, _, files in os.walk(train_dir))
num_val_samples = sum(len(files) for _, _, files in os.walk(val_dir))

# Define batch sizes and image size
train_batch_size = 10
val_batch_size = 10
image_size = 224

# Define paths for training and validation
train_path = train_dir
valid_path = val_dir

# Calculate steps per epoch for training and validation
train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

# Create an ImageDataGenerator for preprocessing
datagen = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input)

# Create image data generators for training, validation, and testing
train_batches = datagen.flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=train_batch_size,
    class_mode='categorical')

valid_batches = datagen.flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=val_batch_size,
    class_mode='categorical')
from tensorflow.keras.optimizers import Adam
# Note: shuffle=False causes the test dataset to not be shuffled
test_batches = datagen.flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=1,
    shuffle=False,
    class_mode='categorical')
mobile = tensorflow.keras.applications.mobilenet.MobileNet()
mobile.summary()
type(mobile.layers)
len(mobile.layers)
# Exclude the last 5 layers of the above model This will include all layers up to and including global_average_pooling2d_1
x = mobile.layers[-6].output
x = Dropout(0.25)(x)
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=predictions)
model.summary()
for layer in model.layers[:-23]:
    layer.trainable = False
# Define Top2 and Top3 Accuracy
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.01), 
              loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

# Get the labels that are associated with each index
print(valid_batches.class_indices)
# Add weights to try to make the model more sensitive to melanoma
# To solve Imbalancedd data problem

class_weights={
    0: 1.0, # akiec
    1: 1.0, # bcc
    2: 1.0, # bkl
    3: 1.0, # df
    4: 3.0, # mel 
    5: 1.0, # nv
    6: 1.0, # vasc
}
import tensorflow as tf
from tensorflow.keras import layers, models

# Define your model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax')  # Assuming 7 classes
])

# Compile the model with both categorical accuracy and top-3 accuracy
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)])

# Display the model summary
model.summary()
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.applications.mobilenet as mobilenet
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Define the base directory
base_dir = 'dataset_dir'

# Define paths for training and validation directories within the base directory
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')

# Get the number of samples in training and validation sets
num_train_samples = sum(len(files) for _, _, files in os.walk(train_dir))
num_val_samples = sum(len(files) for _, _, files in os.walk(val_dir))

# Define batch sizes and image size
train_batch_size = 10
val_batch_size = 10
image_size = 224

# Define paths for training and validation
train_path = train_dir
valid_path = val_dir

# Calculate steps per epoch for training and validation
train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

# Create an ImageDataGenerator for preprocessing
datagen = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input)

# Create image data generators for training, validation, and testing
train_batches = datagen.flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=train_batch_size,
    class_mode='categorical')

valid_batches = datagen.flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=val_batch_size,
    class_mode='categorical')

# Define callbacks
filepath = "model.h5"
from tensorflow.keras.callbacks import ModelCheckpoint

filepath = "model.keras"  # Change the filename to end with .keras
checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1, 
                             save_best_only=True, mode='max')


reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
callbacks_list = [checkpoint, reduce_lr]

# Train the model
history = model.fit(train_batches, steps_per_epoch=int(train_steps), 
                    validation_data=valid_batches,
                    validation_steps=int(val_steps),
                    epochs=10, verbose=1,
                    callbacks=callbacks_list)

model.metrics_names


# Assuming you have already trained a model and have a 'history' object from the training process

# Convert the history object to a dictionary
history_dict = history.history

# Save the dictionary to a pickle file
with open('etc1.pkl', 'wb') as file:
    pickle.dump(history_dict, file)