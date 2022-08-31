
import os
os.chdir('/opt/test')

from findBadmintonCourt import BadmintonCourtFinder
import cv2
import utils
import numpy as np


img = cv2.imread("a-undistort.jpg")
# img = cv2.imread("sample/b7-2.png")

finder = BadmintonCourtFinder()
finder.find(img)

img_transform = cv2.warpPerspective(img, finder.M, (1340 + 100, 610 + 100))
# cv2.imwrite("test2-transform.jpg", img_transform)
utils.printLine(img_transform, BadmintonCourtFinder.standardLines, "test2-transform.jpg")

modelLines = np.array(np.reshape(BadmintonCourtFinder.standardLines, (len(BadmintonCourtFinder.standardLines)*2,2)), dtype='float32')
modelLines = cv2.perspectiveTransform(np.asarray([modelLines]), np.linalg.inv(finder.M))
modelLines = np.array(np.reshape(modelLines[0], (int(len(modelLines[0])/2),4)), dtype="int")
utils.printLine(img, modelLines, "test2-standard.jpg")