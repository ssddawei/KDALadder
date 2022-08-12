import os
import cv2
import csv
import random
import pickle
import numpy as np

print("ok")

img = cv2.imread("a.jpeg")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img2[:,:,0] = 0
img2[:,:,1] = 0
img2 = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(img2, 120, 255, cv2.THRESH_BINARY )
cv2.imwrite("t2.jpg", binary)