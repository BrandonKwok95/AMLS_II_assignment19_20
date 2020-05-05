
#If you want to train model by yourself, you can just run 'train_model.py' in each model file.

#since training model time is too long, here it use trained model to display accurate rate of train-set
#and validation-set. moreover. The test-set label is not open, so the only way to evaluate model in test
#is to upload the .csv file to Kaggle.

#If you want to train model by yourself, you can just run 'train_model.py' in each model file.

import keras
from keras.models import load_model
import os
import pandas as pd
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications import VGG16
from keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

def model_lenet_5():
    # LeNet-5
    model = load_model('./A_LeNet-5/model/LeNet-5.h5')
    batch_size = 16
    image_size = 128
    filenames = os.listdir("./Datasets/train")
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    train_df, validate_df = train_test_split(df, test_size=0.1)
    train_df = train_df.reset_index()
    validate_df = validate_df.reset_index()

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]
    train_df['category'] = train_df['category'].astype(str)
    validate_df['category'] = validate_df['category'].astype(str)

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest',
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        "./Datasets/train/",
        x_col='filename',
        y_col='category',
        class_mode='binary',
        target_size=(image_size, image_size),
        batch_size=batch_size
    )

    validate_datagen = ImageDataGenerator(rescale=1. / 255)
    validate_generator = validate_datagen.flow_from_dataframe(
        validate_df,
        "./Datasets/train/",
        x_col='filename',
        y_col='category',
        class_mode='binary',
        target_size=(image_size, image_size),
        batch_size=batch_size
    )

    loss_A_train, acc_A_train = model.evaluate_generator(train_generator, total_train//batch_size, workers=12)
    print("LeNet-5 Model Train: accuracy = %f  ;  loss = %f " % (acc_A_train, loss_A_train))

    loss_A_val, acc_A_val = model.evaluate_generator(validate_generator, total_validate//batch_size, workers=12)
    print("LeNet-5 Model Validate = %f  ;  loss = %f " % (acc_A_val, loss_A_val))
    return acc_A_train, acc_A_val



def model_vcg16():
    #VCG16
    model = load_model('./B_VCG16/model/VCG16.h5')
    image_size = 224
    batch_size = 16

    filenames = os.listdir("./Datasets/train")
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    train_df, validate_df = train_test_split(df, test_size=0.1)
    train_df = train_df.reset_index()
    validate_df = validate_df.reset_index()


    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]
    train_df['category'] = train_df['category'].astype(str)
    validate_df['category'] = validate_df['category'].astype(str)

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        "./Datasets/train/",
        x_col='filename',
        y_col='category',
        class_mode='binary',
        target_size=(image_size, image_size),
        batch_size=batch_size
    )

    validate_datagen = ImageDataGenerator(rescale=1./255)
    validate_generator = validate_datagen.flow_from_dataframe(
        validate_df,
        "./Datasets/train/",
        x_col='filename',
        y_col='category',
        class_mode='binary',
        target_size=(image_size, image_size),
        batch_size=batch_size
    )

    loss_B_train, acc_B_train = model.evaluate_generator(train_generator, total_train//batch_size, workers=12)
    print("VCG16 Model Train: accuracy = %f  ;  loss = %f " % (acc_B_train, loss_B_train))

    loss_B_val, acc_B_val = model.evaluate_generator(validate_generator, total_validate//batch_size, workers=12)
    print("VCG16 Model Validate = %f  ;  loss = %f " % (acc_B_val, loss_B_val))
    return acc_B_train, acc_B_val


acc_A_train, acc_A_val = model_lenet_5()
acc_B_train, acc_B_val = model_vcg16()

#Here is the result of accurate rate of model(train and validate result)
print('LeNet-5:{},{}; VCG16:{},{};'.format(acc_A_train, acc_A_val,acc_B_train, acc_B_val))
#If you want to get the test result, you can run the test_model in the model file and you will get a csv file,
#you can upload the file to get loss score on Kaggle.