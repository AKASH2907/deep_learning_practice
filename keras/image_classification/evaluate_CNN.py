from keras.models import Model
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.optimizers import Nadam
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications import inception_resnet_v2
from keras.applications import xception
from keras.applications.resnet import ResNet50
from keras.applications.nasnet import NASNetLarge
import numpy as np
from keras.preprocessing import image
from PIL import Image
import pandas as pd
import cv2
import math
import time
from os import listdir
from os.path import join
from sklearn.preprocessing import LabelEncoder
from keras_efficientnets import EfficientNetB5, EfficientNetB0
import argparse


def cnn_model(model_name, img_size):
    """
    Model definition using Xception net architecture
    """
    input_size = (img_size, img_size, 3)

    if model_name == "xception":
        print("Loading Xception wts...")
        baseModel = Xception(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
    elif model_name == "iv3":
        baseModel = InceptionV3(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
    elif model_name == "irv2":
        baseModel = InceptionResNetV2(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
    elif model_name == "resnet":
        baseModel = ResNet50(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
    elif model_name == "nasnet":
        baseModel = NASNetLarge(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
    elif model_name == "ef0":
        baseModel = EfficientNetB0(
            input_size, weights="imagenet", include_top=False 
        )
    elif model_name == "ef5":
        baseModel = EfficientNetB5(
            input_size, weights="imagenet", include_top=False 
        )

    headModel = baseModel.output
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
        headModel
    )
    headModel = Dropout(0.4)(headModel)
    # headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
    #     headModel
    # )
    # headModel = Dropout(0.5)(headModel)
    predictions = Dense(
        5,
        activation="softmax",
        kernel_initializer="he_uniform")(
        headModel
    )
    model = Model(inputs=baseModel.input, outputs=predictions)

    for layer in baseModel.layers:
        layer.trainable = False

    optimizer = Nadam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model


def main():
    start = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model_name", required=True, type=str,
        help="Imagenet model to train", default="xception"
    )
    ap.add_argument(
        "-im_size", "--image_size", required=True, type=int,
        help="Batch size", default=224
    )
    args = ap.parse_args()
    # Read video labels from csv file
    files = listdir("test_frames/")
    files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    labels = np.load("train_label.npy")
    lb = LabelEncoder()
    onehot = lb.fit_transform(labels)
    im_size = args.image_size
    # Loading model weights
    model = cnn_model(model_name=args.model_name, img_size=im_size)
    model.load_weights("trained_wts/" + args.model_name + ".hdf5")
    print("Weights loaded...")

    frame_id = []
    y_predictions = []

    for file in files:
        frame_id+=[file]
        img = image.load_img(join("test_frames", file), target_size=(im_size, im_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = xception.preprocess_input(x)
        predictions = model.predict(x)
        y_predictions.append(lb.inverse_transform([np.argmax(predictions)])[0])

    submission = pd.DataFrame({
        "Frame_ID": frame_id,
        "Emotion": y_predictions
        })
    submission.to_csv("Test.csv", index=False)



if __name__ == "__main__":
    main()
