# img_test.py

import cv2
import os

cwd = os.path.dirname(os.path.realpath(__file__))
pathname = "/images/"

imgs = {}
for filename in os.listdir(cwd + pathname):
    imgs[filename] = cv2.imread(cwd + pathname + filename)
    if filename == "IMG_5255.jpg":
        x = cv2.imread(cwd + pathname + filename)


print(imgs["IMG_5255.jpg"].shape)
xg = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
print(xg.shape)

cv2.imshow("original", x)
cv2.imshow("gray", xg)
cv2.waitKey(0)
cv2.destroyAllWindows()
