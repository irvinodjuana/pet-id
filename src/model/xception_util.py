from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.xception import Xception, preprocess_input

from sklearn.datasets import load_files
import numpy as np
from glob import glob
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow truncated imgs


class Util:
    """Util class for XceptionModel"""

    @staticmethod
    def load_dataset(path):
        """Load a dataset from a path directory for training"""
        num_breeds = 133

        data = load_files(path)
        input_files = np.array(data["filenames"])
        targets = np_utils.to_categorical(np.array(data["target"]), num_breeds)
        return input_files, targets

    @staticmethod
    def path_to_tensor(img_path):
        """Resize colour img and and convert to 4D tensor"""
        img_resize = 224
        # RGB image -> PIL.Image.Image
        img = image.load_img(img_path, target_size=(img_resize, img_resize))
        x = image.img_to_array(img)     # PIL.Image.Image -> 3D Tensor
        x = np.expand_dims(x, axis=0)   # 3D -> 4D Tensor
        return x

    @staticmethod
    def paths_to_tensors(img_paths):
        """Create tensors for collection of images"""
        tensors_list = [Util.path_to_tensor(path)
                        for path in tqdm(img_paths)]
        return np.vstack(tensors_list)

    @staticmethod
    def extract_Xception(file_paths):
        """Manually extract pre-trained Xception bottleneck features"""
        tensors = Util.paths_to_tensors(file_paths).astype('float32')
        preprocessed_input = preprocess_input(tensors)
        return Xception(weights='imagenet',
                        include_top=False).predict(preprocessed_input,
                                                   batch_size=32)

    @staticmethod
    def other_bottleneck_features(path):
        """Get bottleneck features from pre-computed file"""
        bottleneck_features = np.load(path)
        train = bottleneck_features['train']
        valid = bottleneck_features['valid']
        test = bottleneck_features['test']
        return train, valid, test

    @staticmethod
    def evaluate_model(model, model_name, tensors, targets):
        """Returns the prediction accuracy on test data"""
        predicted = [np.argmax(model.predict(np.expand_dims(feature, axis=0)))
                     for feature in tensors]
        test_accuracy = 100 * np.sum(np.array(predicted) == np.argmax(targets,
                                                                      axis=1))
        test_accuracy = test_accuracy / len(predicted)
        print(f'{model_name} accuracy on test data is {test_accuracy}%')
