import numpy as np
import cv2
import glob
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('camera/d/*.png')

for fname in images:
    img = cv2.imread(fname)
    print("readed = ", fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6,9),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (6,9), corners2,ret)
        cv2.imwrite('camera/c/'+os.path.basename(fname)+".jpg", img)
    else:
        print("not found")
        cv2.imwrite('camera/c/'+os.path.basename(fname)+".jpg", img)


img = cv2.imread("a.jpg")
img=cv2.transpose(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
h,  w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# print(gray.shape[::-1])
# print(img.shape[:2])
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

dst=cv2.transpose(dst)
dst=cv2.transpose(dst)
dst=cv2.transpose(dst)
cv2.imwrite('a-undistort.jpg', dst)

#https://wenku.baidu.com/view/0b1b703f68d97f192279168884868762caaebb66.html
