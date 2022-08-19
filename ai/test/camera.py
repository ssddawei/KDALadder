import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*5,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)
print("objp = ", objp)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

img = cv.imread("a-cheese1.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Find the chess board corners
# If found, add object points, image points (after refining them)
corners = np.asarray([
[[184,620]],
[[217,636]],
[[262,657]],
[[320,684]],
[[392,718]],

[[290,615]],
[[330,629]],
[[381,648]],
[[451,671]],
[[536,700]],

[[386,609]],
[[431,623]],
[[489,640]],
[[565,660]],
[[663,684]],

[[478,605]],
[[526,617]],
[[588,632]],
[[670,649]],
[[776,670]],

[[565,600]],

[[619,611]],

[[685,624]],

[[771,639]],

[[880,658]]
], dtype="float32")
ret = True

# ret, corners = cv.findChessboardCorners(gray, (5,5), None)
if ret == True:
    print("corners = ", corners)
    print("corners = ", corners.shape[0])
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    # Draw and display the corners
    cv.drawChessboardCorners(img, (5,5), corners2, ret)
    cv.imwrite('img.jpg', img)
  
h,  w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# print(gray.shape[::-1])
# print(img.shape[:2])
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv.imwrite('img.png', dst)