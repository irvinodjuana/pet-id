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

from PIL import ImageFile

# SSL setup to import pretrained models from keras.applications
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow truncated imgs


# Load file/label strings from a directory of images
def load_dataset(path):
    num_breeds = 133
    data = load_files(path)
    input_files = np.array(data["filenames"])
    targets = np_utils.to_categorical(np.array(data["target"]), num_breeds)
    return input_files, targets


# Resize colour img and and convert to 4D tensor
def path_to_tensor(img_path):
    img_resize = 224
    # RGB image -> PIL.Image.Image
    img = image.load_img(img_path, target_size=(img_resize, img_resize))
    x = image.img_to_array(img)  # PIL.Image.Image -> 3D Tensor
    x = np.expand_dims(x, axis=0)  # 3D -> 4D Tensor
    return x


# Create tensors for collection of images
def paths_to_tensors(img_paths):
    tensors_list = [path_to_tensor(path)
                    for path in tqdm(img_paths)]
    return np.vstack(tensors_list)


# Manually extract pre-trained Xception bottleneck features
def extract_Xception(file_paths):
    tensors = paths_to_tensors(file_paths).astype('float32')
    preprocessed_input = preprocess_input(tensors)
    return Xception(weights='imagenet',
                    include_top=False).predict(preprocessed_input,
                                               batch_size=32)


# Predict with model, labels, img
def predict(dog_breeds, model, img_path):
    tensor = path_to_tensor(img_path)
    feature = Xception(weights='imagenet',
                       include_top=False).predict(preprocess_input(tensor))

    # Get single prediction label (dog breed)
    predicted_vector = model.predict(feature)
    prediction = np.argmax(predicted_vector)
    predict_label = dog_breeds[prediction]

    # Get top N breeds predicted by model with probabilities
    NUM_VALUES = 5
    class_prob = predicted_vector[0]
    top_predict_values = sorted(range(len(class_prob)),
                                key=lambda i: class_prob[i])[-1:-NUM_VALUES:-1]

    for i in top_predict_values:
        print(i, dog_breeds[i], class_prob[i], sep='\t')

    return predict_label


# Build second-level model
def input_branch(input_shape=None):
    size = int(input_shape[2] / 4)

    branch_input = Input(shape=input_shape)
    branch = GlobalAveragePooling2D()(branch_input)
    branch = Dense(size, use_bias=False, kernel_initializer='uniform')(branch)
    branch = BatchNormalization()(branch)
    branch = Activation('relu')(branch)

    return branch, branch_input


# Get bottleneck features from pre-computed file
def other_bottleneck_features(path):
    bottleneck_features = np.load(path)
    train = bottleneck_features['train']
    valid = bottleneck_features['valid']
    test = bottleneck_features['test']
    return train, valid, test


# a function that returns the prediction accuracy on test data
def evaluate_model(model, model_name, tensors, targets):
    predicted = [np.argmax(model.predict(np.expand_dims(feature, axis=0)))
                 for feature in tensors]
    test_accuracy = 100 * np.sum(np.array(predicted) == np.argmax(targets,
                                                                  axis=1))
    test_accuracy = test_accuracy / len(predicted)
    print(f'{model_name} accuracy on test data is {test_accuracy}%')


# Main
def main():
    # Setup parameters
    images_dir = "../data/assets/images/"
    checkpoint_dir = "../data/saved_models/"

    # Process data into files/labels
    # train_files, train_targets = load_dataset(images_dir + "train")
    # valid_files, valid_targets = load_dataset(images_dir + "valid")
    # test_files, test_targets = load_dataset(images_dir + "test")

    dog_breeds = [item.split("/")[-1].split(".")[-1]
                  for item in sorted(glob(images_dir + "train/*"))]

    # dog_breeds = [item[20:-1]
    #               for item in sorted(glob("dog/assets/images/train/*/"))]

    # Obtain bottleneck features for train/valid/test data
    bottleneck_file = "../data/assets/bottleneck_features/ \
                       DogXceptionData.npz"
    train_Xception, valid_Xception, test_Xception = \
        other_bottleneck_features(bottleneck_file)
    # train_Xception = extract_Xception(train_files)
    # valid_Xception = extract_Xception(valid_files)
    # test_Xception = extract_Xception(test_files)

    # Build Xception model
    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    Xception_model.add(Dense(133, activation='softmax'))
    # Xception_model.summary()

    # Compile and train model
    # Xception_model.compile(loss='categorical_crossentropy',
    #                        optimizer='rmsprop', metrics=['accuracy'])
    checkpoint_path = checkpoint_dir + "best_xception_model.hdf5"
    # checkpointer = ModelCheckpoint(filepath=checkpoint_path,
    #                                verbose=1 , save_best_only=True)

    # Xception_history = Xception_model.fit(train_Xception, train_targets,
    #               validation_data = (valid_Xception , valid_targets),
    #               epochs=25, batch_size=20, callbacks=[checkpointer],
    #               verbose=1)

    # Load model weights
    Xception_model.load_weights(checkpoint_path)

    # Evaluate model against testing dataset
    # evaluate_model(Xception_model, "Xception" , test_Xception, test_targets)

    img_path = "/Users/irvinodjuana/Desktop/rosie.png"
    print("\n")
    prediction = predict(dog_breeds, Xception_model, img_path)
    print("\n" + prediction)
    plt.imshow(mpimg.imread(img_path))
    print("\n")


if __name__ == "__main__":
    main()
