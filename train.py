# train.py

from setup import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

cwd = os.path.dirname(os.path.realpath(__file__))
label_dir = "/dataset/labels.csv"
image_dir = "/dataset/aug_images/"

data, targets, filenames = [], [], []

rows = open(cwd + label_dir).read().strip().split("\n")
for row in rows[1:]:
    row = row.split(",")
    (filename, width, height, icon, xmin, ymin, xmax, ymax) = row

    image = cv2.imread(cwd + image_dir + filename)
    xmin = float(xmin) / float(width)
    xmax = float(xmax) / float(width)
    ymin = float(ymin) / float(height)
    ymax = float(ymax) / float(height)
    image = load_img(cwd + image_dir + filename, target_size=(224, 224))
    image = img_to_array(image)

    data.append(image)
    targets.append((xmin, ymin, xmax, ymax))
    filenames.append(filename)

data = np.array(data, dtype="float32") / 225.0
targets = np.array(targets, dtype="float32")

split = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

print("saving testing filenames...")
f = open(cwd + "/output/test_images.txt", "w")
f.write("\n".join(testFilenames))
f.close()

vgg = VGG16(
    weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))
)
vgg.trainable = False

flatten = vgg.output
flatten = Flatten()(flatten)

bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)

model = Model(inputs=vgg.input, outputs=bboxHead)

opt = Adam(lr=config.INIT_LR)
model.compile(loss="mse", optimizer=opt)
print(model.summary())

print("training bounding box regressor...")
H = model.fit(
    trainImages,
    trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    verbose=1,
)

print("saving object detector model....")
model.save(cwd + "/output/detector.h5", save_format="h5")

N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(cwd + "/output/plot.png")
