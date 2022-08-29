import enum
import cv2
import numpy as np
import os
import math
import csv
import itertools
import time
import random
from multiprocessing import Pool

from test3 import HoughBundler

if __name__ != '__main__':
  exit()

# os.chdir('E:\\git\\kdaladder\\ai\\test')
os.chdir('/opt/test')

def flood_fill(field, x ,y, old, new):
  # we need the x and y of the start position, the old value,
  # and the new value
  # the flood fill has 4 parts
  # firstly, make sure the x and y are inbounds
  if x < 0 or x >= len(field[0]) or y < 0 or y >= len(field):
      return
  # secondly, check if the current position equals the old value
  if field[y][x] != old:
      return
  
  # thirdly, set the current position to the new value
  field[y][x] = new
  # fourthly, attempt to fill the neighboring positions
  flood_fill(field, x+1, y, old, new)
  flood_fill(field, x-1, y, old, new)
  flood_fill(field, x, y+1, old, new)
  flood_fill(field, x, y-1, old, new)

def get_orientation(line):
    x1, y1, x2, y2 = line
    if x1 > x2:
      x1, y1 = line[2:]
      x2, y2 = line[0:2]
    orientation = math.atan2(((y2 - y1)), ((x2 - x1)))
    return math.degrees(orientation)

def get_orientation_abs(line):
    orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
    return math.degrees(orientation)

def line_intersection(line1, line2):
    xdiff = (line1[0] - line1[2], line2[0] - line2[2])
    ydiff = (line1[1] - line1[3], line2[1] - line2[3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return [0.0, 0.0]
        raise Exception('lines do not intersect')

    d = (det(line1[:2], line1[2:]), det(line2[:2], line2[2:]))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]
def get_cross_point(lines):
    ret = []
    for i in lines[:int(len(lines)/2)]:
        for j in lines[int(len(lines)/2):]:
            ret.append(line_intersection(i, j))
    return ret
    # return [
    #     line_intersection(lines[0], lines[2]),
    #     line_intersection(lines[0], lines[3]),
    #     line_intersection(lines[1], lines[2]),
    #     line_intersection(lines[1], lines[3]),
    # ]

def printLine(lines, filename):
  r = 255
  g = 0
  b = 0
  line_color = [r, g, b]
  line_thickness = 3
  dot_color = [0, 255, 0]
  dot_size = 3
  line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8) 
  # #讲检测的直线叠加到原图
  for line in lines:
      x1, y1, x2, y2 = line
      line_color = [b, g, r]
      cv2.line(line_img, (x1, y1), (x2, y2), line_color, line_thickness)
      cv2.circle(line_img, (x1, y1), dot_size, [0, 255, 0], -1)
      cv2.circle(line_img, (x2, y2), dot_size, [0, 0, 255], -1)

      if g >= 255:
        b += 20
        r = 0
      elif b >= 255:
        r += 20
        g = 0
      elif r >= 255:
        g += 20
        b = 0
  final = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

  cv2.imwrite(filename, final)

def isLinePixel(img, point):
  T = 20
  Threshold = 50
  diffThreshold = 20

  cols = img.shape[1]
  rows = img.shape[0]
  x, y = point
  if x < T or cols - x <= T:
    return 0
  if y < T  or rows - y <= T:
    return 0

  value = img[y][x]
  if value < Threshold:
    return 0

  topValue = img[y-T][x]
  lowerValue = img[y+T][x]
  leftValue = img[y][x-T]
  rightValue = img[y][x+T]

  # if ((value - leftValue > parameters.diffThreshold) && (value - rightValue > parameters.diffThreshold))
  # {
  #   return GlobalParameters().fgValue;
  # }

  # if ((value - topValue > parameters.diffThreshold) && (value - lowerValue > parameters.diffThreshold))
  # {
  #   return GlobalParameters().fgValue;
  # }
  if ((int(value) - leftValue > diffThreshold) and (int(value) - rightValue > diffThreshold)):
    return 255

  if ((int(value) - topValue > diffThreshold) and (int(value) - lowerValue > diffThreshold)):
    return 255

  return 0

def getLinePixel(img):
  ret = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) 
  for y in range(img.shape[0]):
    for x in range(img.shape[1]):
      ret[y][x] = isLinePixel(img, [x, y])
  return ret

def getLinePixel2(img, T):
  # T = 30
  # T = int(max(img.shape[1],img.shape[0])/500)

  T = int(T)
  Threshold = 50
  diffThreshold = 10

  ret = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) 
  matrix = np.asarray(img, dtype=np.int)
  leftImg = np.roll(matrix, T, axis=1)
  rightImg = np.roll(matrix, -T, axis=1)
  topImg = np.roll(matrix, T, axis=0)
  bottomImg = np.roll(matrix, -T, axis=0)

  lowest = (matrix - Threshold).clip(min=0,max=1)

  subLeftImg = ((matrix - leftImg) - diffThreshold).clip(min=0,max=1)
  subRightImg = ((matrix - rightImg) - diffThreshold).clip(min=0,max=1)
  subHImg = np.bitwise_and(subLeftImg, subRightImg)

  subTopImg = ((matrix - topImg) - diffThreshold).clip(min=0,max=1)
  subBottomImg = ((matrix - bottomImg) - diffThreshold).clip(min=0,max=1)
  subVImg = np.bitwise_and(subTopImg, subBottomImg)

  ret = np.bitwise_or(subHImg, subVImg)
  ret = np.bitwise_and(lowest, ret)
  return np.array(ret * 255, np.uint8)

def computeStructureTensorElements(image):

  # Mat floatImage, dx, dy;
  # image.convertTo(floatImage, CV_32F);
  floatImage = np.float32(image)
  # GaussianBlur(floatImage, floatImage, Size(5,5), 0);
  floatImage = cv2.GaussianBlur(floatImage, (5, 5), 0)
  # cv2.imwrite("test2-GaussianBlur.jpg", floatImage)
  # Sobel(floatImage, dx, CV_32F, 1, 0, parameters.gradientKernelSize);
  # Sobel(floatImage, dy, CV_32F, 0, 1, parameters.gradientKernelSize);
  dx = cv2.Sobel(floatImage, -1, 1, 0)
  dy = cv2.Sobel(floatImage, -1, 0, 1)
  # multiply(dx, dx, dx2);
  # multiply(dx, dy, dxy);
  # multiply(dy, dy, dy2);
  dx2 = dx
  dy2 = dy
  # dx2 = cv2.multiply(dx, dx)
  dxy = cv2.multiply(dx, dy)
  # dy2 = cv2.multiply(dy, dy)
  # # Mat kernel = Mat::ones(parameters.kernelSize, parameters.kernelSize, CV_32F);
  # # filter2D(dx2, dx2, -1, kernel);
  # # filter2D(dxy, dxy, -1, kernel);
  # # filter2D(dy2, dy2, -1, kernel);
  kernel = np.ones((3, 3), np.float32)
  dx2 = cv2.filter2D(dx2, -1, kernel)
  dxy = cv2.filter2D(dxy, -1, kernel)
  dy2 = cv2.filter2D(dy2, -1, kernel)
  # dx2 = dx
  # dy2 = dy
  # dxy = cv2.multiply(dx, dy)
  return (dx2, dxy, dy2)

def filterLinePixels(binaryImage, luminanceImage):

  # Mat dx2, dxy, dy2;
  # computeStructureTensorElements(luminanceImage, dx2, dxy, dy2);
  # Mat outputImage(binaryImage.rows, binaryImage.cols, CV_8UC1, Scalar(GlobalParameters().bgValue));
  dx2, dxy, dy2 = computeStructureTensorElements(luminanceImage)
  cv2.imwrite("test2-dx.jpg", dx2)
  cv2.imwrite("test2-dy.jpg", dy2)
  cv2.imwrite("test2-dxy.jpg", dxy)
  outputImage = np.zeros(binaryImage.shape, np.uint)
  # for (unsigned int x = 0; x < binaryImage.cols; ++x)
  # {
  for x in range(binaryImage.shape[1]):
    # for (unsigned int y = 0; y < binaryImage.rows; ++y)
    # {
    for y in range(binaryImage.shape[0]):
      value = binaryImage[y][x]
      if value == 255:
        # Mat t(2, 2, CV_32F);
        t = np.zeros((2,2), np.float32)
        # t.at<float>(0, 0) = dx2.at<float>(y,x);
        # t.at<float>(0, 1) = dxy.at<float>(y,x);
        # t.at<float>(1, 0) = dxy.at<float>(y,x);
        # t.at<float>(1, 1) = dy2.at<float>(y,x);
        t[0][0] = dx2[y][x]
        t[0][1] = dxy[y][x]
        t[1][0] = dxy[y][x]
        t[1][1] = dy2[y][x]
        # Mat l;
        # eigen(t, l);
        retval, eigenvalues, eigenvectors = cv2.eigen(t)
        # if (l.at<float>(0,0) > 4* l.at<float>(0,1))
        # {
        if eigenvalues[0][0] > 4 * eigenvalues[1][0]:
        #   outputImage.at<uchar>(y,x) = GlobalParameters().fgValue;
          outputImage[y][x] = 255
  return outputImage

def point_to_line_distance(point, line):
  # Point2f pointOnLine = getPointOnLineClosestTo(point);
      # Point2f n;
      # float c;
      # toImplicit(n, c);
          # n = perpendicular(v);
            # return cv::Point2f(-v.y, v.x);
          # c = n.dot(u);
  point = np.asarray(point)
  u = [line[1], line[0]]
  v = [line[3], line[2]]
  n = [-v[0], v[1]]
  c = np.dot(n, u)
      # float q = c - n.dot(point);
      # return point - q*n;
  q = c - np.dot(n, point)
  pointOnLine = point - q * n
  # return distance(point, pointOnLine);
      # float dx = p1.x - p2.x;
      # float dy = p1.y - p2.y;
      # return sqrt(dx*dx + dy*dy);
  dx = point[1] - pointOnLine[1]
  dy = point[0] - pointOnLine[0]
  return math.sqrt(dx*dx + dy*dy)

def point_to_line_distance2(point, line):
  p3 = np.array(point)
  p1 = np.array(line[0:2])
  p2 = np.array(line[2:4])
  return np.linalg.norm(abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1))

def point_to_line_distance3(point, line):
  dis1 = math.sqrt(abs(point[0] - line[0]) ** 2 + abs(point[1] - line[1]) ** 2)
  dis2 = math.sqrt(abs(point[0] - line[2]) ** 2 + abs(point[1] - line[3]) ** 2)
  return min(dis1, dis2)
# print(point_to_line_distance2([0,2],[1,1,2,0]))

def getClosePointsMatrix(line, binaryImage):
  distanceThreshold = 10
#   Mat M = Mat::zeros(0, 2, CV_32F);
  # M = np.zeros((0, 2), np.float32)
  M = []

#   Mat image = rgbImage.clone(); // debug

#   for (unsigned int x = 0; x < binaryImage.cols; ++x)
#   {
  for x in range(binaryImage.shape[1]):
#     for (unsigned int y = 0; y < binaryImage.rows; ++y)
#     {
    for y in range(binaryImage.shape[0]):
#       if (binaryImage.at<uchar>(y, x) == GlobalParameters().fgValue)
#       {
      if binaryImage[y][x] == 255:
#         float distance = line.getDistance(Point2f(x, y));
        distance = point_to_line_distance2([x, y], line)
#         if (distance < parameters.distanceThreshold)
#         {
        if distance < distanceThreshold:
# //          drawPoint(Point2f(x, y), image, Scalar(255,0,0));
#           Mat point = Mat::zeros(1, 2, CV_32F);
#           point.at<float>(0, 0) = x;
#           point.at<float>(0, 1) = y;
#           M.push_back(point);
          point = np.zeros((1, 2), np.float32)
          point[0][0] = x
          point[0][1] = y
          M.append(point)
  return np.array(M, np.float32)

def getRefinedParameters(line, binaryImage):
  # Mat A = getClosePointsMatrix(line, binaryImage, rgbImage);
  A = getClosePointsMatrix(line, binaryImage)
  # Mat X = Mat::zeros(1, 4, CV_32F);
  # X = np.zeros((1, 4), np.float32)
  # fitLine(A, X, DIST_L2, 0, 0.01, 0.01);
  X = cv2.fitLine(A, cv2.DIST_L2, 0, 0.01, 0.01)
  # Point2f v(X.at<float>(0,0), X.at<float>(0,1));
  # Point2f p(X.at<float>(0,2), X.at<float>(0,3));
  # return Line(p, v);
  return X


def refineLine(line, binary):
  T = 4
  lineImg = np.zeros(binary.shape, np.uint8)
  x1, y1, x2, y2 = line
  cv2.line(lineImg, (x1, y1), (x2, y2), 255, T)
  refinelineImg = cv2.bitwise_and(lineImg, binary)
  coords = np.flip(np.column_stack(np.where(refinelineImg == 255)), axis = 1)

  lineImg2 = np.zeros(binary.shape, np.uint8)
  for i in coords:
    cv2.circle(lineImg2, i, 2, 255, -1)

  vx, vy, x, y = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)
  m = 1000
  rline = [x[0]-m*vx[0], y[0]-m*vy[0], x[0]+m*vx[0], y[0]+m*vy[0]]
  return np.array(rline, np.int)

def merge_lines(lines):
  degMax = 1
  disMax = 5

  groups = []
  for line in lines:
    foundMatch = 0
    for group in groups:
      for gline in group:
        deg1 = get_orientation(line)
        deg2 = get_orientation(gline)
        if abs(deg1 - deg2) < degMax:
          dis = point_to_line_distance2(line[0:2], gline)
          if dis < disMax:
            group.append(line)
            foundMatch = 1
            break
      if foundMatch: continue
    if not foundMatch:
        groups.append([line])
  result = []
  for i in groups:
    vx, vy, x, y = cv2.fitLine(np.array(i).reshape((-1,2)), cv2.DIST_L2, 0, 0.01, 0.01)
    m = 300
    line = [x-m*vx[0], y-m*vy[0], x+m*vx[0], y+m*vy[0]]
    result.append(line)
  return np.array(result, np.int).reshape((-1,4))

def merge_lines2(lines):
  degMax = 2
  disMax = 2

  groups = []
  for line in lines:
    foundMatch = 0
    for group in groups:
      for gline in group:
        deg1 = get_orientation(line)
        deg2 = get_orientation(gline)
        if abs(deg1 - deg2) < degMax:
          if deg1 > 0:
            dis1 = point_to_line_distance2([0, img.shape[0]], line)
            dis2 = point_to_line_distance2([0, img.shape[0]], gline)
          else:
            dis1 = point_to_line_distance2([img.shape[1], img.shape[0]], line)
            dis2 = point_to_line_distance2([img.shape[1], img.shape[0]], gline)
          if abs(dis1 - dis2) < disMax:
            group.append(line)
            foundMatch = 1
            break
      if foundMatch: continue
    if not foundMatch:
        groups.append([line])
  result = []
  for i in groups:
    vx, vy, x, y = cv2.fitLine(np.array(i).reshape((-1,2)), cv2.DIST_L2, 0, 0.01, 0.01)
    m = 300
    line = [x-m*vx[0], y-m*vy[0], x+m*vx[0], y+m*vy[0]]
    result.append(line)
  return np.array(result, np.int).reshape((-1,4))

def generateGroundMask(img):
  img_small = cv2.resize(img, (int(img.shape[1]/10), int(img.shape[0]/10)))
  Z = img_small.reshape((-1,3))

  # convert to np.float32
  Z = np.float32(Z)

  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 5
  ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

  # Now convert back into uint8, and make original image
  label_unique = np.unique(label, return_counts=1)
  target_label = 0
  for idx, val in enumerate(label_unique[1]):
    if label_unique[1][target_label] < val: target_label = idx
  cetner2 = np.zeros((5,3), np.float32)
  cetner2[target_label] = (255,255,255)
  # cetner2=center
  cetner2 = np.uint8(cetner2)
  res = cetner2[label.flatten()]
  res2 = res.reshape((img_small.shape))

  kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  res2 = cv2.dilate(res2, kernel2, iterations=2)
  res2 = cv2.resize(res2, (img.shape[1], img.shape[0]))
  ret, res2 = cv2.threshold(res2, 160, 255, cv2.THRESH_BINARY )
  return res2

def polar2cer(line):
  rho, theta = line
  a = np.cos(theta) 
  b = np.sin(theta)
  x0 = a * rho 
  y0 = b * rho
  return [
    int(x0 + 1000 * (-b)),
    int(y0 + 1000 * (a)),
    int(x0 - 1000 * (-b)),
    int(y0 - 1000 * (a))
  ]

def extend_line_w0(line, width):
    # print('extend: ', line)
    deg = get_orientation(line)
    if (45 < deg <= 90):
        return line
    x1, y1, x2, y2 = line
    topRate = (x2 - x1) / (x1 - 0)
    bottomRate = (x2 - x1) / (width - x2)
    if topRate > 0:
        ny1 = y1 - (y2 - y1) / topRate 
    else:
        ny1 = y1
    if bottomRate > 0:
        ny2 = y2 + (y2 - y1) / bottomRate 
    else:
        ny2 = y2

    return np.array([0, ny1, width, ny2], np.int)
# 读入图片
img = cv2.imread("sample/b7-2.png")
# img = cv2.imread("a-undistort.jpg")
# img = cv2.resize(img, (1280, int(1280*img.shape[0]/img.shape[1])), interpolation = cv2.INTER_LINEAR_EXACT)
print("img shape = ", img.shape)

# gMask = generateGroundMask(img)
# cv2.imwrite("test2-gMask.jpg", gMask)

# img = cv2.bitwise_and(img, gMask)
# cv2.imwrite("test2-masked.jpg", img)


# 中值滤波，去噪
# img = cv2.medianBlur(img, 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("test2-gray.jpg", gray)


  # Mat imgYCbCr;
  # cvtColor(frame, imgYCbCr, CV_RGB2YCrCb);
  # Mat luminanceChannel(frame.rows, frame.cols, CV_8UC1);
  # const int from_to[2] = {0, 0};
  # mixChannels(&frame, 1, &luminanceChannel, 1, from_to, 1);

ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

cr = np.array(ycrcb[:,:,1], np.int)
cb = np.array(ycrcb[:,:,2], np.int)
white = cv2.inRange(cr - cb, -30, 30)

lum2 = np.zeros(gray.shape, dtype=np.uint8)
lum2[:,:] = ycrcb[:,:,0]

lum2 = cv2.bitwise_and(lum2, lum2, mask=white)

lum2 = cv2.medianBlur(lum2, 3)

# print("ycrcb = ", ycrcb)
# print("lum = ", lum2)
cv2.imwrite("test2-lum.jpg", cv2.cvtColor(lum2, cv2.COLOR_GRAY2BGR))
# img2 = cv2.imread("test2-pixel.jpg", cv2.IMREAD_GRAYSCALE)
binary = getLinePixel2(lum2, max(img.shape[:2])/60)
binary_ex = getLinePixel2(lum2, max(img.shape[:2])/600)
binary = cv2.bitwise_or(binary, binary_ex)
cv2.imwrite("test2-pixel.jpg", binary)
# filter = filterLinePixels(img2, lum2)
# cv2.imwrite("test2-filter.jpg", filter)
print("img2 = ", binary)

# https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv
nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(binary)
sizes = stats[:, -1]
# the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
# you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
sizes = sizes[1:]
nb_blobs -= 1

# minimum size of particles we want to keep (number of pixels).
# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
min_size = 100  

# output image with only the kept components
binary2 = np.zeros((binary.shape), np.uint8)
blobDrawing = np.zeros((img.shape), np.uint8)
# for every component in the image, keep it only if it's above min_size
for blob in range(nb_blobs):
    if sizes[blob] >= min_size:
        # see description of im_with_separated_blobs above
        binary2[im_with_separated_blobs == blob + 1] = 255

        coords = np.flip(np.column_stack(np.where(im_with_separated_blobs == blob+1)), axis = 1)
        (x,y),radius = cv2.minEnclosingCircle(coords)
        # area = cv2.contourArea(coords)
        if radius > max(binary.shape[:2])/20 :#and area < (binary.shape[0]*binary.shape[0])/20:
            blobDrawing[im_with_separated_blobs == blob + 1] = (
              random.randint(0,255),random.randint(0,255),random.randint(0,255))

cv2.imwrite("test2-blobDrawing.jpg", blobDrawing)
exit()
# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# # binary2 = cv2.dilate(binary2, kernel2, iterations=2)
# binary2 = cv2.erode(binary2, kernel2, iterations=int(max(binary.shape[:2])/120/4))

cv2.imwrite("test2-binary2.jpg", binary2)

binary2 = cv2.resize(binary2,np.int32((binary2.shape[1]/2,binary2.shape[0]/2)))
binary = cv2.resize(binary,np.int32((binary.shape[1]/2,binary.shape[0]/2)))

(contours,hierarchy) = cv2.findContours(binary2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]

binary3 = np.zeros((binary.shape), np.uint8)

# Draw contours
drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    # if hierarchy[i][2] > 0:
    #   continue
    (x,y),radius = cv2.minEnclosingCircle(contours[i])
    if radius < max(binary.shape[:2])/20 :
      continue
    color = random.randint(0,256)
    if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
        cv2.drawContours(drawing, contours, i, (0, 0, color), -1)
    else:
        cv2.drawContours(drawing, contours, i, (0, color, 0), -1)

    # color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
    # cv2.drawContours(drawing, contours, i, color, -1)
cv2.imwrite("test2-contours.jpg", drawing)
for i in range(len(contours)):
  # area = cv2.contourArea(contour)
  # perimeter = cv2.arcLength(contour,True)
  (x,y),radius = cv2.minEnclosingCircle(contours[i])
  if radius > max(binary.shape[:2])/2 :
    cv2.drawContours(binary3, contours, i, 255, 2)

# binary3 = cv2.bitwise_and(binary2, binary3)
cv2.imwrite("test2-binary3.jpg", binary3)

exit()


canny = cv2.Canny(binary3, threshold1=80, threshold2=200, apertureSize=7)
cv2.imwrite("test2-canny.jpg", canny)

lines = cv2.HoughLinesP(canny, 
  rho=1, 
  theta=np.pi / 360, 
  threshold=10, 
  minLineLength=10, 
  maxLineGap=10)
lines = [i[0] for i in lines]

# lines = cv2.HoughLines(canny, 
#   1, 
#   np.pi / 180, 
#   150, 
#   None, 
#   0,
#   0)
# lines = [polar2cer(i[0]) for i in lines]

# lines = [extend_line_w0(i, img.shape[1]) for i in lines]

print("lines = ", lines)
printLine(lines, "test2-lines.jpg")

rLines = [refineLine(i,canny) for i in lines]
printLine(rLines, "test2-rLines.jpg")

# mlines = merge_lines(rLines)

# printLine(mlines, "test2-merge.jpg")

mlines = merge_lines2(rLines)

printLine(mlines, "test2-merge2.jpg")



#######################
#######################

# 从左下角和右下角分别向中心点寻找最近的两条线

leftLines = []
rightLines = []

for line in mlines:
    deg = get_orientation(line)
    if deg > 0:
        leftLines.append(line)
    else:
        rightLines.append(line)

def distance_by_arrow(arrow, line):
  crossPoint = line_intersection(arrow, line)
  dis = math.sqrt(
      math.pow(arrow[0] - crossPoint[0], 2) + 
      math.pow(arrow[1] - crossPoint[1], 2)
    )
  return dis

# leftLines = sorted(leftLines, key=lambda i: 
#     distance_by_arrow([0, img.shape[0], img.shape[1], 0],i))
# rightLines = sorted(rightLines, key=lambda i: 
#     distance_by_arrow([img.shape[1], img.shape[0], 0, 0],i))

leftLines = sorted(leftLines, key=lambda i: 
    point_to_line_distance3([0, img.shape[0]],i))
rightLines = sorted(rightLines, key=lambda i: 
    point_to_line_distance3([img.shape[1], img.shape[0]],i))

leftLines = leftLines[0:15]
rightLines = rightLines[0:15]


printLine(leftLines, "test2-left.jpg")
printLine(rightLines, "test2-right.jpg")

standardLines = []
with open('standard.csv') as csvfile:
    rows = list(csv.reader(csvfile, delimiter=','))
    for row in rows[1:]:
        standardLines.append(np.array(row, dtype=np.int))

print("standardLines = ", np.asarray(standardLines))

def calcM(crossPoint):
    crossPoint = np.array([
        [j for j in i] for i in crossPoint
    ], dtype='float32')
    sp = np.array([
        [[*j, 0] for j in i] for i in Scross_point
    ], dtype='float32')
    print("sp = ", sp)
    print("crossPoint = ", crossPoint)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(sp.reshape((1,-1,3)), crossPoint.reshape((1,-1,2)), gray.shape[::-1], None, None)
    R = cv2.Rodrigues(rvecs[0])[0]
    t = tvecs[0]
    Rt = np.concatenate([R,t], axis=-1) # [R|t]
    P = np.matmul(mtx,Rt) # A[R|t]
    return np.linalg.inv(np.delete(P, 2, 1))
    
def calcM3(crossPoint):
    crossPoint = np.array([
        [j for j in i] for i in crossPoint
    ], dtype='float32')
    sp = np.array([
        [[*j, 0] for j in i] for i in Scross_point
    ], dtype='float32')
    print("sp = ", sp)
    print("crossPoint = ", crossPoint)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(sp.reshape((1,-1,3)), crossPoint.reshape((1,-1,2)), gray.shape[::-1], None, None)
    K = mtx
    T = tvecs[0]
    R = cv2.Rodrigues(rvecs[0])[0]
    Hr = np.matmul(np.matmul(K, np.linalg.inv(R)), np.linalg.inv(K))
    C = np.matmul(-np.linalg.inv(R), T)
    Cz = C[2]
    l = np.array([[1,0],[0,1],[0,0]])
    Ht = np.concatenate((l, np.matmul(-K, C/Cz)), axis=1)
    return np.linalg.inv(np.matmul(Ht, Hr))
    
def calcM4(modelCrossPoint, crossPoint):
  # M,_ = cv2.findHomography(sp.reshape((1,-1,3)), crossPoint.reshape((1,-1,2)))
  M,_ = cv2.findHomography(np.float32(modelCrossPoint), np.float32(crossPoint))
  return np.linalg.inv(M)
  
def calcM2(crossPoint):
    return cv2.getPerspectiveTransform(
        np.array(crossPoint[0], dtype='float32'),
        np.array(Scross_point[0], dtype='float32'))
def calcStandardLines(modelCrossPoint, crossPoint):
    M = calcM4(modelCrossPoint, crossPoint)
    # print("M = ", M)
    srcLines = np.array(np.reshape(standardLines, (len(standardLines)*2,2)), dtype='float32')

    sLines = cv2.perspectiveTransform(np.asarray([srcLines]), np.linalg.inv(M))
    sLines = np.array(np.reshape(sLines[0], (int(len(sLines[0])/2),4)), dtype="int")
    return sLines
def lineOffscreenLen(line, maxWidth, maxHeight):
    # 计算超出屏幕的部分百分比
    x1, y1, x2, y2 = line
    rate = 0
    if x1 < 0 or x2 < 0:
        rate += -min(x1, x2) / abs(x2-x1) 
    elif y1 < 0 or y2 < 0:
        rate += -min(y1, y2) / abs(y2-y1) 
    if x1 > maxWidth or x2 > maxWidth:
        rate += (max(x1, x2) - maxWidth) / abs(x2-x1) 
    elif y1 > maxHeight or y2 > maxHeight:
        rate += (max(y1, y2) - maxHeight) / abs(y2-y1) 
    return rate


bundler = HoughBundler(min_distance=10,min_angle=2)
def line_intersect(s_lineset, lineset):
    score = [0] * len(s_lineset)
    notmatch = 0
    for l2 in lineset:
        matched = 0
        for idx, l1 in enumerate(s_lineset):
            d1 = get_orientation_abs(l1)
            d2 = get_orientation_abs(l2)
            dis = bundler.get_distance(l1, l2)
            if abs(d1 - d2) < 1 and dis < 10:
            # if abs(d1 - d2) < 2:
                score[idx] += 1
                matched = 1
                break
        if not matched:
            notmatch += 1
    # print("score = ", score)
    # print("notmatch = ", notmatch)
    # score.append(notmatch)
    # return np.var(score)
    return notmatch
        

def scoreCalc(modelCrossPoint, crossPoint):
  sLines = calcStandardLines(modelCrossPoint, crossPoint)  
  lineImg = np.zeros((binary.shape), np.uint8)
  # for l in sLines[0:2]:
  #   cv2.line(lineImg, l[0:2], l[2:4], 255, 2)
  # for l in sLines[12:14]:
  #   cv2.line(lineImg, l[0:2], l[2:4], 255, 2)
  for l in sLines:
    cv2.line(lineImg, l[0:2], l[2:4], 255, 4)
  
  crossImg = cv2.bitwise_and(lineImg, canny)
  score = len(np.where(crossImg == 255)[0])
  return score

def findBiggestRect(leftLines, rightLines):
   
  # 对左右两边的线条进行组合
  leftLinesCom = [[leftLines[i], leftLines[i+1]] for i in range(len(leftLines)-1)]
  rightLinesCom = [[rightLines[i], rightLines[i+1]] for i in range(len(rightLines)-1)]

  # 从小到大的组合两个数组，目的是优先检测低索引的组合
  lineSet = itertools.product(range(len(leftLinesCom)), range(len(rightLinesCom)))
  lineSet = sorted(lineSet, key = lambda i: max(i))
  lineSet = [(leftLinesCom[i[0]], rightLinesCom[i[1]]) for i in lineSet]

  imgCrossSet = list(map(lambda i:
    [line_intersection(j[0], j[1]) for j in itertools.product(i[0], i[1])], lineSet
  ))

  areaSet = list(map(lambda i:(i[0], cv2.contourArea(np.float32(i[1]))), enumerate(imgCrossSet)))
  areaSet = sorted(areaSet, key = lambda i: -i[1])
  return lineSet[areaSet[0][0]]

def fitModel(preMatchLines, modelLines, imgLines):
  '''
  preMatchLines: 已有的配对
  modelLines: 要配对的model lines
  imgLines: 要配对的图片 lines

  return:
    matchLines: 配对成功的 lines
    standardLines: 结果 lines
  '''
  pmLeftLines, pmRightLines = list(preMatchLines[0])
  pLeftLines, pRightLines = list(preMatchLines[1])
  mLeftLines, mRightLines = list(modelLines)
  leftLines, rightLines = list(imgLines)

  mlength = len(pmLeftLines) + len(mLeftLines)
  
  # 对左右两边的线条进行组合
  leftLinesCom = list(itertools.combinations(range(len(leftLines)), mlength))
  leftLinesCom = sorted(leftLinesCom, key = lambda i: max(i))
  leftLinesCom = [[leftLines[j] for j in i] for i in leftLinesCom]

  rightLinesCom = list(itertools.combinations(range(len(rightLines)), mlength))
  rightLinesCom = sorted(rightLinesCom, key = lambda i: max(i))
  rightLinesCom = [[rightLines[j] for j in i] for i in rightLinesCom]

  # 从小到大的组合两个数组，目的是优先检测低索引的组合
  lineSet = itertools.product(range(len(leftLinesCom)), range(len(rightLinesCom)))
  lineSet = sorted(lineSet, key = lambda i: max(i))
  lineSet = [(leftLinesCom[i[0]], rightLinesCom[i[1]]) for i in lineSet]

  # 获取imglines交点
  imgCrossSet = list(map(lambda i:
    [line_intersection(j[0], j[1]) for j in itertools.product(i[0], i[1])], lineSet
  ))
  # 获取modellines交点
  modelCross = [line_intersection(j[0], j[1]) for j in itertools.product(pmLeftLines + mLeftLines, pmRightLines + mRightLines)]
  

  # 计算分数
  # scores = [(idx, scoreCalc(modelCross, i)) for idx,i in enumerate(imgCrossSet)]
  scores = []
  best = 0
  print("imgCrossSet size = ", len(imgCrossSet))
  for idx,i in enumerate(imgCrossSet):
    score = scoreCalc(modelCross, i)
    scores.append((idx, score))
    if score > best:
      best = score
      calcedStardardLines = calcStandardLines(modelCross, imgCrossSet[int(idx)])
      print("found best: ", idx)
      print("found score: ", best)
      printLine(list(calcedStardardLines) + list(np.reshape(lineSet[idx],(-1,4))), "test2-best.jpg")
      time.sleep(1)
     
  scores = sorted(scores, key=lambda i: -i[1])
  calcedStardardLines = calcStandardLines(modelCross, imgCrossSet[int(scores[0][0])])
  return calcedStardardLines
    
ModelLines =[
# 左上角矩形
    [
        # standardLines[-10],
        standardLines[-8],
        standardLines[-5],
    ],
    [
        # standardLines[0],
        standardLines[2],
        standardLines[4],
    ]
]

# binggestRect = findBiggestRect(leftLines, rightLines)
# printLine(np.array(binggestRect).reshape((-1,4)), "test2-biggestRectLines.jpg")
# exit()
calcedStardardLines = fitModel([[[],[]],[[],[]]], ModelLines, [leftLines, rightLines])
printLine(calcedStardardLines, "test2-standard.jpg")