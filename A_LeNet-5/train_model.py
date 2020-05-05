from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPool2D,Flatten,Dense,Dropout
import PIL
import tensorflow.keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import build_form

image_size = 128
batch_size= 32
df = build_form.create_df()
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
    "../Datasets/train/",
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(image_size, image_size),
    batch_size=batch_size
)

validate_datagen = ImageDataGenerator(rescale=1. / 255)
validate_generator = validate_datagen.flow_from_dataframe(
    validate_df,
    "../Datasets/train/",
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(image_size, image_size),
    batch_size=batch_size
)

model = Sequential([
    Convolution2D(32, 3, 3, input_shape=(128, 128, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Convolution2D(64, 3, 3, input_shape=(128, 128, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model_LeNet = model.fit_generator(
    train_generator, steps_per_epoch=100, epochs=100, verbose=1, validation_data=validate_generator, validation_steps=100,

)

model_LeNet.save('./model/LeNet-5.h5')
#visualization the training process
def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
    axs[0].plot(range(1, len(model_history.history[acc]) + 1), model_history.history[acc])
    axs[0].plot(range(1, len(model_history.history[val_acc]) + 1), model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history[acc]) + 1), len(model_history.history[acc]) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    plt.savefig('./model/training_process_LeNet-5.png')

plot_model_history(model_LeNet)