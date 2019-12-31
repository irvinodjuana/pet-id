from keras.utils import np_utils
from keras.preprocessing import image

from keras.applications.xception import Xception, preprocess_input

from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential

from sklearn.datasets import load_files
import numpy as np
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from xception_util import Util

# SSL setup to import pretrained models from keras.applications
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class XceptionModel:
    """
    Model for dog image breed prediction using transfer learning
    on Xception-based pretrained CNN
    """

    def __init__(self):
        """Initialize Xception model class"""
        self.model = None
        self.num_breeds = 133
        self.dog_breeds = None
        self.imgs_dir = None


    def learn(self, imgs_dir, bottleneck_file=None, checkpoint_dir=None):
        """Learn/train model from images dataset"""
        self.model = self.build_model()
        self.imgs_dir = imgs_dir

        # Load and format images from dataset for model use
        train_files, train_targets = Util.load_dataset(self.imgs_dir + "train")
        valid_files, valid_targets = Util.load_dataset(self.imgs_dir + "valid")
        test_files, test_targets = Util.load_dataset(self.imgs_dir + "test")

        self.dog_breeds = [item.split("/")[-1].split(".")[-1]
                           for item in sorted(glob(self.imgs_dir + "train/*"))]

        # Extract bottleneck features from pre-computed file or manually
        if bottleneck_file:
            train_Xception, valid_Xception, test_Xception = \
                Util.other_bottleneck_features(bottleneck_file)
        else:
            train_Xception = Util.extract_Xception(train_files)
            valid_Xception = Util.extract_Xception(valid_files)
            test_Xception = Util.extract_Xception(test_files)

        # Compile model and setup checkpointer
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop', metrics=['accuracy'])

        # Save checkpoints if provided and train model
        if checkpoint_dir:
            checkpoint_path = checkpoint_dir + "best_xception_model.hdf5"
            checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                                           verbose=1, save_best_only=True)
            self.model.fit(train_Xception, train_targets,
                           validation_data=(valid_Xception, valid_targets),
                           epochs=25, batch_size=20, callbacks=[checkpointer],
                           verbose=1)

        else:
            self.model.fit(train_Xception, train_targets,
                           validation_data=(valid_Xception, valid_targets),
                           epochs=25, batch_size=20, verbose=1)

    def load_pretrained_model(self, weights_path, breeds_path):
        """
        Load weights from pretrained model (.hdf5) into current model
        Creates list of dog breeds from .txt file
        """
        self.model = self.build_model()

        # Load weights from .hdf5 file
        self.model.load_weights(weights_path)
        self.dog_breeds = []
        # Load breeds line by line from text file
        with open(breeds_path, "r") as file:
            for breed in file:
                self.dog_breeds.append(breed.strip())

    def predict(self, img_path, num_values=5):
        """
        Predict breed(s) and probabilities based on image file
        Returns list of tuples: [(breed, probability), ...]
        """
        if not self.model:
            return None     # Model not yet initialized

        tensor = Util.path_to_tensor(img_path)
        feature = Xception(weights='imagenet',
                           include_top=False).predict(preprocess_input(tensor))

        # Get top N breeds predicted by model with probabilities
        predicted_vector = self.model.predict(feature)
        class_prob = predicted_vector[0]
        top_predict_values = sorted(range(len(class_prob)),
                                    key=lambda i: class_prob[i])[-num_values:]
        top_predict_values = top_predict_values[::-1]
        predictions = [(self.dog_breeds[i], class_prob[i])
                       for i in top_predict_values]

        return predictions
    
    def detect_dog(self, img_path):
        """Detect whether or not image contains dog"""
        # img needs to be resized to 299, base xception input shape
        img = preprocess_input(Util.path_to_tensor(img_path, img_resize=299))
        base_model = Xception(weights='imagenet')
        prediction = np.argmax(base_model.predict(img))
        return bool(prediction >= 151 and prediction <= 268)


    def build_model(self):
        """Build the model"""
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
        model.add(Dense(self.num_breeds, activation='softmax'))
        return model
