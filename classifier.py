import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras

def show_result(img_path, cat_percent, dog_percent):
    img = mpimg.imread(img_path)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    _ = plt.imshow(img)
    ax.set_title("This image is %.2f% cat and %.2f% dog."
        % (cat_percent, dog_percent))
    plt.show()


def main():
    test_path = sys.argv[1]
    
    image_size = (180, 180)
    path = 'model/classifier.h5'
    model = keras.models.load_model(path)

    img = keras.preprocessing.image.load_img(
    test_path, target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = predictions[0]
    dog_percent = score * 100
    cat_percent = 100 - dog_percent
    print(
        "This image is %.2f percent cat and %.2f percent dog."
        % (cat_percent, dog_percent)
    )

    show_result(test_path, cat_percent, dog_percent)

if __name__ == "__main__":
    main()