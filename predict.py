# predict.py

from setup import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os

cwd = os.path.dirname(os.path.realpath(__file__))
image_dir = "/dataset/aug_images/"

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--input",
    required=True,
    help="path to input image/text file of image filenames",
)
args = vars(ap.parse_args())

filetype = mimetypes.guess_type(args["input"])[0]
image_paths = [args["input"]]

if filetype == "text/plain":
    filenames = open(args["input"]).read().strip().split("\n")
    image_paths = []
    for file in filenames:
        p = cwd + image_dir + file
    image_paths.append(p)

print("loading object detector...")
model = load_model(cwd + "/output/detector.h5")

for image_path in image_paths:
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]
    (xmin, ymin, xmax, ymax) = preds

    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    xmin = int(xmin * w)
    ymin = int(ymin * h)
    xmax = int(xmax * w)
    ymax = int(ymax * h)

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow("output", image)
    cv2.waitKey(0)
