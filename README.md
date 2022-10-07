import numpy as np
import os
from keras.layers import Flatten
import pandas as pd

def load_data(DATA_PATH):
    X = []
    Y = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram_img = np.load(file_path[-1]) # (n_bins, n_frames, 1)
            x_train.append(spectrogram_img)
            labels = np.load(file_path[2])
    X = np.array(X)
    Y = np.array(Y)
    X = X[..., np.newaxis] 
    return X, Y

def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, Y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

import keras.preprocessing.image
import keras.utils
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks
from keras.models import  model_from_json,Sequential
from keras.models import Model as Md
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import *
from skimage import color
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def Model:
    
    images = keras.layers.Input(x_train.shape[1:])
    
    x = keras.layers.Conv2D(filters=16, kernel_size=[3, 3], padding='same')(images)
    block = keras.layers.BatchNormalization()(x)
    block = keras.layers.Activation("relu")(block)

    # ResNet block 2a
    block2 = keras.layers.Conv2D(filters=16, kernel_size=[3, 3], padding='same')(block)
    block2 = keras.layers.BatchNormalization()(block2)
    block2 = keras.layers.Activation("relu")(block2)

    # ResNet block 3a
    block3 = keras.layers.Conv2D(filters=16, kernel_size=[3, 3], padding='same')(block2)
    block3 = keras.layers.BatchNormalization()(block3)
    block3 = keras.layers.Activation("relu")(block3)

    #inio Squeeze and Excitation 1a
    sq = keras.layers.GlobalAveragePooling2D()(block3)
    sq = keras.layers.Reshape((1,1,16))(sq)
    sq = keras.layers.Dense(units=16,activation="relu")(sq)
    sq = keras.layers.Dense(units=16,activation="sigmoid")(sq)
    block = keras.layers.multiply([block3,sq])

    #Res block
    x = keras.layers.Conv2D(filters=16, kernel_size=[3, 3], padding='same')(images)
    x = keras.layers.BatchNormalization()(x)

    #final Output
    net = keras.layers.add([x,block])
    net = keras.layers.Activation("relu")(net)

    net = Flatten()(net)
    net = Dropout(0.5)(net) 
    x = Dense(6)(net)
    net = keras.layers.Activation("relu")(x)
    
    return model

def train(x_train, y_train, learning_rate, batch_size, epochs):
    
    checkpointer = keras.callbacks.ModelCheckpoint(filepath = 'cnn_from_scratch_fruits.hdf5', verbose = 1, save_best_only = True)
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    train_model = model.fit(x_train, y_train, batch_size, learning_rate, epochs, callbacks = [checkpointer,earlystopper], shuffle=True)
    return train_model

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    model = keras.models.Model(inputs=images,outputs=net)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

#preprocesser

import librosa
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image, ImageOps

class Loader:
    
    def __init__(self):
        self = self
        
    def load(self, img_path):
        im = Image.open(img_path)
        return im    

class Img_process:
    
    def __init__(self):
        self = self
    
    def Img_process(im):    
        #im = Image.open(img_path)
        im1 = np.array(ImageOps.grayscale(im))
        resized_img = cv2.resize(im1, dsize = [128, 128], interpolation = cv2.INTER_AREA)
        norm_img = cv2.normalize(resized_img, None, 0, 100, cv2.NORM_MINMAX)
    
        return norm_img

class Label_array:
    
    #image,label
    def __init__(self):
        self = self
        
    def parse_file_name(self, file_path: str):
        filename = file_path.split("/")[-1]
        label = int(filename.split("-")[1])
        return label, filename
    
    def labeling(self, label, processed_img):
        labeled_img = append.processed_img        
        return labeled_img  

class Saver:
    """ saver is responsible to save features """

    def __init__(self, feature_save_dir):
        self.feature_save_dir = feature_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)

    
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path

class PreprocessingPipeline:
    
    def __init__(self):
        self._img_process = None
        self.labeler = None
        self.parcer = None
        self.saver = None
        self._loader = None
        
    def loader(self):
        return self._loader
 
    
    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")
    
    def _process_file(self, file_path):
        im = self.loader.load(file_path)
        img_array = self._img_process.process(im)
        parced_fn = self.parcer.parse_file_name(file_path)
        labeled_img = self.labeler.labeling(parced_fn, img_array)
        save_path = self.saver.save_feature(labeled_img, file_path)     
        
    

if __name__ == "__main__":

    IMAGE_SAVE_DIR = "Desktop/1"
    FILES_DIR = "img_processed"

    # instantiate all objects
    loader = Loader()
    imgprocess = Img_process()
    labeler = Label_array()
    saver = Saver(IMAGE_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline._img_process = imgprocess
    preprocessing_pipeline.labeler = labeler
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)

#spectrogram 

import librosa
import pandas
import matplotlib.pyplot as plt
import os
import pickle
import librosa
import numpy as np
import json
import librosa.display
import IPython.display as ipd

class Loader:
    """Loader is responsible for loading an audio file."""

    def __init__(self, sample_rate, mono):
        self.sample_rate = sample_rate
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              mono=self.mono)[0]
        return signal   

class MelSpectrogramExtractor:
    """ extracts mel spectograms (in db) from a time series signal """
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length
        
    def extract(self, signal):
        spec = librosa.feature.melspectrogram(y = signal)
        mel_spectrogram = librosa.amplitude_to_db(spec, ref = np.max)
        return mel_spectrogram

class Saver:
    """ saver is responsible to save features """

    def __init__(self, feature_save_dir):
        self.feature_save_dir = feature_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)

    
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path

class PreprocessingPipeline:

    def __init__(self):
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self._loader = None
        #self._num_expected_samples = None

    
    def loader(self):
        return self._loader

    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")
        
    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        feature = self.extractor.extract(signal)
        save_path = self.saver.save_feature(feature, file_path)

if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    SAMPLE_RATE = 16000
    MONO = True

    SPECTROGRAMS_SAVE_DIR = "Desktop/spectrograms"
    FILES_DIR = "demo"

    # instantiate all objects
    loader = Loader(SAMPLE_RATE, MONO)
    mel_spectrogram_extractor = MelSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    saver = Saver(SPECTROGRAMS_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.extractor = mel_spectrogram_extractor
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)
    
