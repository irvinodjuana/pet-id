from xception_model import XceptionModel
from glob import glob


# Main
def main():
    # Setup parameters
    data_dir = "../../data/"
    images_dir = data_dir + "assets/images/"
    checkpoint_dir = data_dir + "saved_models/"

    weights_path = data_dir + "saved_models/best_xception_model.hdf5"
    breeds_path = data_dir + "assets/dog_breeds.txt"
    bottleneck_features_path = data_dir + "assets/bottleneck_features/DogXceptionData.npz"

    xception_model = XceptionModel()

    # xception_model.learn(images_dir,
    #                      bottleneck_file=bottleneck_features_path,
    #                      checkpoint_dir=checkpoint_dir)
    
    xception_model.load_pretrained_model(weights_path, breeds_path)

    img_path1 = "/Users/irvinodjuana/Desktop/rosie.png"
    img_path2 = "/Users/irvinodjuana/Downloads/cat2.jpeg"

    # Test breed predictions
    predictions = xception_model.predict(img_path1)
    print(predictions)
    # Test dog detection
    print("Rosie is a dog: ", xception_model.detect_dog(img_path1))     # True
    print("Cat is a dog: ", xception_model.detect_dog(img_path2))       # False

    # count = 0
    # dogs = 0

    # for file in glob(images_dir + "test/**/*.jpg")[:20]:
    #     count += 1
    #     if xception_model.detect_dog(file):
    #         dogs += 1
    
    # print(f"Percentage of dogs detected in train: {dogs}/{count}")


if __name__ == "__main__":
    main()
