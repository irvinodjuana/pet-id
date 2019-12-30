from xception_model import XceptionModel


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
    xception_model.load_pretrained_model(weights_path, breeds_path)

    # xception_model.learn(images_dir,
    #                      bottleneck_file=bottleneck_features_path,
    #                      checkpoint_dir=checkpoint_dir)

    img_path1 = "/Users/irvinodjuana/Desktop/rosie.png"
    img_path2 = "/Users/irvinodjuana/Downloads/dog.jpeg"
    predictions = xception_model.predict(img_path1)
    print(predictions)


if __name__ == "__main__":
    main()
