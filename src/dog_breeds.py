from keras.utils import np_utils
from keras.preprocessing import image

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

from PIL import ImageFile
from sklearn.datasets import load_files

import numpy as np
from glob import glob
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Load file/label strings from a directory of images
def load_dataset(path):
    num_breeds = 133
    data = load_files(path)
    input_files = np.array(data["filenames"])
    targets = np_utils.to_categorical(np.array(data["target"]), num_breeds)
    return input_files, targets


# Resize colour img and and convert to 4D tensor
def img_path_to_tensor(img_path):
    img_resize = 224
    # RGB image -> PIL.Image.Image
    img = image.load_img(img_path, target_size=(img_resize, img_resize))
    im_tensor = image.img_to_array(img)  # PIL.Image.Image -> 3D Tensor
    im_tensor = np.expand_dims(im_tensor, axis=0)  # 3D -> 4D Tensor
    return im_tensor


# Create tensors for collection of images
def paths_to_tensors(img_paths):
    tensors_list = [img_path_to_tensor(path)
                    for path in tqdm(img_paths)]
    return np.vstack(tensors_list)


# Build pre-trained resnet50 model
def extract_ResNet50(file_paths):
    tensors = paths_to_tensors(file_paths).astype('float32')
    prep_input = preprocess_input_resnet50(tensors)
    model = ResNet50(weights='imagenet',
                     include_top=False).predict(prep_input, batch_size=32)
    return model


# Build pre-trained vgg19 model
def extract_VGG19(file_paths):
    tensors = paths_to_tensors(file_paths).astype('float32')
    prep_input = preprocess_input_vgg19(tensors)
    model = VGG19(weights='imagenet',
                     include_top=False).predict(prep_input, batch_size=32)
    return model


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

    # Pre-process data for Keras
    # train_tensors = paths_to_tensors(train_files).astype('float32')/255
    # test_tensors = paths_to_tensors(test_files).astype('float32')/255
    # valid_tensors = paths_to_tensors(valid_files).astype('float32')/255

    train_vgg19 = extract_VGG19(train_files)
    valid_vgg19 = extract_VGG19(valid_files)
    test_vgg19 = extract_VGG19(test_files)
    print("VGG19 shape", train_vgg19.shape[1:])

    train_resnet50 = extract_ResNet50(train_files)
    valid_resnet50 = extract_ResNet50(valid_files)
    test_resnet50 = extract_ResNet50(test_files)
    print("Resnet50 shape", train_resnet50.shape[1:])

    


if __name__ == "__main__":
    main()
