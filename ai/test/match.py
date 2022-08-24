import cv2
import numpy as np
import os
import math
import csv
from multiprocessing import Pool

from test3 import HoughBundler
from calcM import calcM

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
  line_thickness = 2
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

# 读入图片
img = cv2.imread("bd2.png")

# 中值滤波，去噪
img = cv2.medianBlur(img, 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 阈值分割得到二值化图片
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ret, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY )
# ret, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY )

# flood_fill(binary, 0, 0, 0, 2)
cv2.imwrite("test2-a.jpg", binary)
exit
# 膨胀操作
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_clo = cv2.dilate(binary, kernel2, iterations=2)

# edges = cv2.Canny(binary, 55, 110)
edges = cv2.Canny(binary, threshold1=50, threshold2=200, apertureSize=7)
cv2.imwrite("test2-edges.jpg", edges)

#HoughLinesP方法判断哪些边缘是直线
lines = cv2.HoughLinesP(edges, 
  rho=1, 
  theta=np.pi / 180, 
  threshold=90, 
  minLineLength=80, 
  maxLineGap=50)
printLine([i[0] for i in lines], "test2-lines.jpg")

# Merge

bundler = HoughBundler(min_distance=5,min_angle=1)
mlines = bundler.process_lines(lines)
mlines = [i[0] for i in mlines]
# print("mlines = ", mlines)

printLine(mlines, "test2-merge.jpg")

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

leftLines = sorted(leftLines, key=lambda i: 
    distance_by_arrow([0, img.shape[0], img.shape[1], 0],i))
rightLines = sorted(rightLines, key=lambda i: 
    distance_by_arrow([img.shape[1], img.shape[0], 0, 0],i))

leftLines = leftLines[0:10]
rightLines = rightLines[0:5]

# 每边使用两条线进行配对
lineSet = []
for iL1 in range(len(leftLines) - 1):
    for iL2 in range(iL1+1, len(leftLines)):
        # if bundler.get_distance(leftLines[iL1], leftLines[iL2]) < 10: continue
        for iR1 in range(len(rightLines) - 1):
            for iR2 in range(iR1+1, len(rightLines)):
                # if bundler.get_distance(rightLines[iR1], rightLines[iR2]) < 10: continue
                lineSet.append([leftLines[iL1], leftLines[iL2], rightLines[iR1], rightLines[iR2]])

lineSet = sorted(lineSet, key=lambda i: 
    distance_by_arrow([0, img.shape[0], img.shape[1], 0],i[0]) +
    distance_by_arrow([0, img.shape[0], img.shape[1], 0],i[1]) +
    distance_by_arrow([img.shape[1], img.shape[0], 0, 0],i[2]) +
    distance_by_arrow([img.shape[1], img.shape[0], 0, 0],i[3])
  )

printLine(leftLines, "test2-left.jpg")
printLine(rightLines, "test2-right.jpg")
# printLine(leftLines[0:5], "test2-left.jpg")
# printLine(rightLines[0:5], "test2-right.jpg")

# 得到四个交点，用这四个点计算映射矩阵
corssPointSet = list(map(get_cross_point, lineSet))
print("corssPointSet = ", np.asarray(corssPointSet))

standardLines = []
with open('standard.csv') as csvfile:
    rows = list(csv.reader(csvfile, delimiter=','))
    for row in rows[1:]:
        standardLines.append(np.array(row, dtype=np.int))

print("standardLines = ", np.asarray(standardLines))

SLineSet =[
# 左上角矩形
    [
        standardLines[-10],
        standardLines[-8],
        standardLines[0],
        standardLines[2],
    ],
# 左上角最大矩形
    # [
    #     standardLines[-10],
    #     standardLines[-6],
    #     standardLines[0],
    #     standardLines[4],
    # ],
# 左上角两个矩形
    # [
    #     # standardLines[-2],
    #     # standardLines[-4],
    #     standardLines[-7],
    #     standardLines[-6],
    #     # standardLines[1],
    #     # standardLines[3],
    #     standardLines[3],
    #     standardLines[4],
    # ],
]
print("SLineSet = ", np.asarray(SLineSet))
Scross_point = list(map(get_cross_point, SLineSet))

Scross_point = np.array(Scross_point, dtype='float32')
print("Scross_point = ", Scross_point)

# M =  [[1.08433481e+03 0.00000000e+00 9.23899379e+02]
#  [0.00000000e+00 1.44395319e+02 4.92158299e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# M =  [[ 1.86969389e-01  1.25285042e-01 -4.39081715e+02]
#  [ 2.08050410e-01 -1.60330178e+00  7.64957189e+02]
#  [ 1.88111183e-05 -2.15842946e-03  1.00000000e+00]]
def calcM(crossPoint):
    # crossPoint = np.array([
    #     [[j] for j in i] for i in crossPoint
    # ], dtype='float32')
    # sp = np.array([
    #     [[*j, 0] for j in i] for i in Scross_point
    # ], dtype='float32')
    # print("sp = ", sp)
    # print("crossPoint = ", crossPoint)
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(sp[:1], [crossPoint], gray.shape[::-1], None, None)
    # R = cv2.Rodrigues(rvecs[0])[0]
    # t = tvecs[0]
    # Rt = np.concatenate([R,t], axis=-1) # [R|t]
    # P = np.matmul(mtx,Rt) # A[R|t]
    # return np.linalg.inv(np.delete(P, 2, 1))
    # print("crossPoint = ", crossPoint)
    # print("Scross_point = ", Scross_point[0])
    return cv2.getPerspectiveTransform(
        np.array(crossPoint, dtype='float32'),
        np.array(Scross_point[0], dtype='float32'))
def calcStandardLines(crossPoint):
    M = calcM(crossPoint)
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
        

def scoreCalc(crossPoint):

    sLines = calcStandardLines(crossPoint)

    # offscreen = []
    # for l in sLines:
    #     offscreen.append(lineOffscreenLen(l, img.shape[1], img.shape[0]))

    # offscreen = sum(offscreen)/len(offscreen)

    return line_intersect(sLines, [i[0] for i in lines]) # + offscreen
    # print("offscreen = ", offscreen)
    # print("offscreen = ", sum(offscreen)/len(offscreen))

    sLines = np.array([[l] for l in sLines], dtype="int")
    # print("sLines = ", sLines)
    # print("lines = ", mlines)
    # print("sLines = ", len(sLines))
    # print("lines = ", len(mlines))
    sLines = np.concatenate((sLines, mlines))
    linesMerged = bundler.process_lines(sLines)
    # return (offscreen * 100 + 1) * (abs(len(mlines) - len(linesMerged)))
    return abs(len(mlines) - len(linesMerged)) + offscreen

# 对 M 进行评分
# 1. 在画面内的部分占比
# 2. 线条去重后，多出来的线条数（越少越匹配）
scores = []

print("corssPointSet size = ", len(corssPointSet))
def f(param):
  offset, crossPointSubset = param
  subScores = []
  print("corssPointSet size = ", len(crossPointSubset))
  for idx, cp in enumerate(crossPointSubset):
    score = scoreCalc(cp)
    subScores.append([int(idx + offset),score])
  return subScores

# THREAD = 10
# SPLIT = 45
# with Pool(THREAD) as p:
#   arr = []
#   for i in range(THREAD):
#     arr.append((i * SPLIT, corssPointSet[i * SPLIT: (i+1) * SPLIT]))
#   scores = np.concatenate(p.map(f, arr))
        
# for idx, cp in enumerate(corssPointSet[:300]):
#     score = scoreCalc(cp)
#     scores.append([idx,score])
#     # if score == 0:
#     #     break
#     if score < 10:
#         break

# scores = sorted(scores, key=lambda i: i[1])
# print("scores = ", scores)
# print("idx = ", scores[0][0])
# calcedStardardLines = calcStandardLines(corssPointSet[int(scores[0][0])])

# l = [leftLines[5],leftLines[11],rightLines[3],rightLines[5]]
l = [
    # [leftLines[2],leftLines[5],rightLines[1],rightLines[6]],
    [leftLines[1],leftLines[3],rightLines[0],rightLines[2]]
]
c = list(map(get_cross_point, l))
c = [get_cross_point(i) for i in l]
c = [
    # get_cross_point([leftLines[1],leftLines[3],rightLines[1],rightLines[3]]),
    # get_cross_point([leftLines[3],leftLines[4],rightLines[3],rightLines[4]])
    get_cross_point(l[0]),
    # get_cross_point(l[1])
    # get_cross_point([leftLines[0],leftLines[2],leftLines[3],leftLines[10],rightLines[0],rightLines[2],rightLines[3],rightLines[6]])
]

printLine(
    l[0] + SLineSet[0], 
    "test2-inputlines.jpg")
# c = get_cross_point(leftLines[1:4] + rightLines[1:4])
calcedStardardLines = calcStandardLines(c)
print("score = ", scoreCalc(c))
print("calcedStardardLines = ", calcedStardardLines)

# ？？
# 考虑尺寸
# 考虑使用更多的点位，提高 M 的准确率
# 尝试使用 caliCamera 提供多组数据
# 尝试一下 3x3 的9个点进行 caliCamera

printLine(calcedStardardLines, "test2-standard.jpg")