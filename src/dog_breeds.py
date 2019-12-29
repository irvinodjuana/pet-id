from keras.utils import np_utils
from keras.preprocessing import image

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow truncated imgs

from sklearn.datasets import load_files

import numpy as np
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# SSL setup to import pretrained models from keras.applications
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


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


# Build pre-trained vgg19 model
def extract_VGG19(file_paths):
    tensors = paths_to_tensors(file_paths).astype('float32')
    preprocessed_input = preprocess_input_vgg19(tensors)
    return VGG19(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)


# Build pre-trained resnet50 model
def extract_ResNet50(file_paths):
    tensors = paths_to_tensors(file_paths).astype('float32')
    preprocessed_input = preprocess_input_resnet50(tensors)
    return ResNet50(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)


# Extract bottleneck features from vgg19 and resnet 50
def extract_features(img_path):
    tensor = path_to_tensor(img_path)
    bottleneck_vgg19 = VGG19(weights='imagenet', 
        include_top=False).predict(preprocess_input_vgg19(tensor))
    bottleneck_resnet50 = ResNet50(weights='imagenet', 
        include_top=False).predict(preprocess_input_resnet50(tensor))
    return [bottleneck_vgg19, bottleneck_resnet50]


# Predict with model, labels, img
def predict(dog_breeds, model, img_path):
    predicted_vector = model.predict(extract_features(img_path))
    return dog_breeds[np.argmax(predicted_vector)]


# Build second-level model
def input_branch(input_shape=None):
    size = int(input_shape[2] / 4)

    branch_input = Input(shape=input_shape)
    branch = GlobalAveragePooling2D()(branch_input)
    branch = Dense(size, use_bias=False, kernel_initializer='uniform')(branch)
    branch = BatchNormalization()(branch)
    branch = Activation('relu')(branch)

    return branch, branch_input


# Display image
def display_img(img_path):
    plt.imshow(mpimg.imread(img_path))


# Main
def main():
    # Setup parameters
    images_dir = "../data/assets/images/"

    # Process data into files/labels
    train_files, train_targets = load_dataset(images_dir + "train")
    valid_files, valid_targets = load_dataset(images_dir + "valid")
    test_files, test_targets = load_dataset(images_dir + "test")

    dog_breeds = [item.split("/")[-1].split(".")[-1]
                  for item in sorted(glob(images_dir + "train/*"))]
    
    # dog_breeds = [item[20:-1] for item in sorted(glob("dog/assets/images/train/*/"))]

    # train_vgg19 = extract_VGG19(train_files)
    # valid_vgg19 = extract_VGG19(valid_files)
    # test_vgg19 = extract_VGG19(test_files)
    # print("Extracted vgg19")
    # print("VGG19 shape", train_vgg19.shape[1:])

    # train_resnet50 = extract_ResNet50(train_files)
    # valid_resnet50 = extract_ResNet50(valid_files)
    # test_resnet50 = extract_ResNet50(test_files)
    # print("Extracted resnet50")
    # print("Resnet50 shape", train_resnet50.shape[1:])

    vgg19_branch, vgg19_input = input_branch(input_shape=(7, 7, 512))
    resnet50_branch, resnet50_input = input_branch(input_shape=(7, 7, 2048))

    concat_branches = Concatenate()([vgg19_branch, resnet50_branch])
    net = Dropout(0.3)(concat_branches)
    net = Dense(640, use_bias=False, kernel_initializer='uniform')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dropout(0.3)(net)
    net = Dense(133, kernel_initializer='uniform', activation='softmax')(net)

    model = Model(inputs=[vgg19_input, resnet50_input], outputs=[net])
    # model.summary()

    # Compile and fit model
    checkpoint_path = "../data/saved_models/bestmodel.hdf5"
    NUM_EPOCHS = 10
    BATCH_SIZE = 4

    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
    #               metrics=['accuracy'])
    # checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1,
    #                                save_best_only=True)
    # model.fit([train_vgg19, train_resnet50], train_targets, 
    #           validation_data=([valid_vgg19, valid_resnet50], valid_targets),
    #           epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpointer],
    #           verbose=1)

    model.load_weights(checkpoint_path)

    # from sklearn.metrics import accuracy_score

    # predictions = model.predict([test_vgg19, test_resnet50])
    # breed_predictions = [np.argmax(prediction) for prediction in predictions]
    # breed_true_labels = [np.argmax(true_label) for true_label in test_targets]
    # print('Test accuracy: %.4f%%' % (accuracy_score(breed_true_labels, breed_predictions) * 100))

    img_path = "../data/assets/images/test/054.Collie/Collie_03794.jpg"
    print(predict(dog_breeds, model, img_path))
    plt.imshow(mpimg.imread(img_path))



if __name__ == "__main__":
    main()
