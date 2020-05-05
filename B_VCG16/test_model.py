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

def test():
    model = load_model('./model/VCG16.h5')
    test_filenames = os.listdir("../Datasets/test1")
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    nb_samples = test_df.shape[0]
    batch_size = 16
    image_size = 224

    test_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        "../Datasets/test1/",
        x_col='filename',
        y_col=None,
        class_mode=None,
        batch_size=batch_size,
        target_size=(image_size, image_size),
        shuffle=False
    )

    predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
    threshold = 0.5
    test_df['category'] = predict
    sub_df = test_df.copy()
    sub_df['id'] = sub_df['filename'].str.split('.').str[0]
    sub_df['label'] = sub_df['category']
    sub_df.drop(['filename', 'category'], axis=1, inplace=True)
    sub_df.to_csv('./model/sub_VCG16.csv', index=False)