# img_test.py

import cv2
import os
import xml.etree.ElementTree as ET

cwd = os.path.dirname(os.path.realpath(__file__))
pathname = "/images/"
annot_path = "/annotations/"
file_name = "IMG_5210"

# imgs = {}
# for filename in os.listdir(cwd + pathname):
#     imgs[filename] = cv2.imread(cwd + pathname + filename)
#     if filename == "IMG_5210.jpg":
#         ex = cv2.imread(cwd + pathname + filename)


def find_boxes(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    boxes = []
    for icon in root.iter("object"):
        boxes.append([])
        for coord in icon[4]:
            boxes[-1].append(int(coord.text) // 4)
    return boxes


boxes = find_boxes(cwd + annot_path + file_name + ".xml")

ex = cv2.imread(cwd + pathname + file_name + ".jpg")
gray_scale = cv2.cvtColor(ex, cv2.COLOR_BGR2GRAY)
original_dim = gray_scale.shape
resized_dim = (gray_scale.shape[1] // 4, gray_scale.shape[0] // 4)
gs_resized = cv2.resize(gray_scale, resized_dim)

for box in boxes:
    cv2.rectangle(gs_resized, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
cv2.imshow("gray", gs_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
