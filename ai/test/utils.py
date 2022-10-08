
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

def printLine(img, lines, filename):
  r = 255
  g = 0
  b = 0
  line_color = [r, g, b]
  line_thickness = 1
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



def point_to_line_distance2(point, line):
  p3 = np.array(point)
  p1 = np.array(line[0:2])
  p2 = np.array(line[2:4])
  return np.linalg.norm(abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1))

def point_to_line_distance3(point, line):
  dis1 = math.sqrt(abs(point[0] - line[0]) ** 2 + abs(point[1] - line[1]) ** 2)
  dis2 = math.sqrt(abs(point[0] - line[2]) ** 2 + abs(point[1] - line[3]) ** 2)
  return min(dis1, dis2)

def generateGroundMask(img):
  img_small = cv2.resize(img, (int(img.shape[1]/10), int(img.shape[0]/10)))
  Z = img_small.reshape((-1,3))

  # convert to np.float32
  Z = np.float32(Z)

  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 10
  ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

  # Now convert back into uint8, and make original image
  label_unique = np.unique(label, return_counts=1)
  target_label = 0
  for idx, val in enumerate(label_unique[1]):
    if label_unique[1][target_label] < val: target_label = idx
  cetner2 = np.zeros((K,3), np.float32)
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
