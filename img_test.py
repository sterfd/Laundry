# img_test.py

import cv2
import os

cwd = os.path.dirname(os.path.realpath(__file__))
pathname = "/images/"

imgs = {}
for filename in os.listdir(cwd + pathname):
    imgs[filename] = cv2.imread(cwd + pathname + filename)

print(imgs.keys())
