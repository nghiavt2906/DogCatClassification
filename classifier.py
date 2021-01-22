import joblib
import tensorflow as tf
from tensorflow import keras

def main():
    image_size = (180, 180)
    path = 'model/classifier.h5'
    model = keras.models.load_model(path)
    # model = joblib.load(path)

    img = keras.preprocessing.image.load_img(
    "PetImages/Cat/6779.jpg", target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent cat and %.2f percent dog."
        % (100 * (1 - score), 100 * score)
    )

if __name__ == "__main__":
    main()