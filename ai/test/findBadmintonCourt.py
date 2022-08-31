import enum
import cv2
import numpy as np
import os
import math
import csv
import itertools
import time
import random
import utils
from multiprocessing import Pool

class BadmintonCourtFinder:
  DEBUG = 0
  standardLines = []
  with open('standard.csv') as csvfile:
      rows = list(csv.reader(csvfile, delimiter=','))
      for row in rows[1:]:
          standardLines.append(np.array(row, dtype=np.int))
  
  def __init__(self) -> None:
      pass

  def refineLine(self, line, binary):
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

  def merge_lines(self, lines):
    degMax = 1
    disMax = 5

    groups = []
    for line in lines:
      foundMatch = 0
      for group in groups:
        for gline in group:
          deg1 = utils.get_orientation(line)
          deg2 = utils.get_orientation(gline)
          if abs(deg1 - deg2) < degMax:
            dis = utils.point_to_line_distance2(line[0:2], gline)
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

  def merge_lines2(self, lines):
    degMax = 2
    disMax = 2

    groups = []
    for line in lines:
      foundMatch = 0
      for group in groups:
        for gline in group:
          deg1 = utils.get_orientation(line)
          deg2 = utils.get_orientation(gline)
          if abs(deg1 - deg2) < degMax:
            if deg1 > 0:
              dis1 = utils.point_to_line_distance2([0, self.img.shape[0]], line)
              dis2 = utils.point_to_line_distance2([0, self.img.shape[0]], gline)
            else:
              dis1 = utils.point_to_line_distance2([self.img.shape[1], self.img.shape[0]], line)
              dis2 = utils.point_to_line_distance2([self.img.shape[1], self.img.shape[0]], gline)
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

  def removeSmallObject(self, binary):

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
    # blobDrawing = np.zeros((binary.shape), np.uint8)
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above

            coords = np.flip(np.column_stack(np.where(im_with_separated_blobs == blob+1)), axis = 1)
            (x,y),radius = cv2.minEnclosingCircle(coords)
            # area = cv2.contourArea(coords)
            if radius > max(binary.shape[:2])/10 :#and area < (binary.shape[0]*binary.shape[0])/20:
              binary2[im_with_separated_blobs == blob + 1] = 255
              # blobDrawing[im_with_separated_blobs == blob + 1] = (
              #   random.randint(0,255),random.randint(0,255),random.randint(0,255))

    # cv2.imwrite("test2-blobDrawing.jpg", blobDrawing)

    return binary2

  def getLinePixel(self, img, T):
    T = int(T)
    Threshold = 50
    diffThreshold = 20

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

  def getLinePixel2(self, img, T):
    '''
    快一丢丢，但是边沿缺失一个 T
    '''
    # T = 30
    # T = int(max(img.shape[1],img.shape[0])/500)

    TS = time.time()
    T = int(T)
    Threshold = 50
    diffThreshold = 20

    ret = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) 
    matrix = np.asarray(img, dtype=np.int)
    
    lowest = (matrix - Threshold).clip(min=0,max=1)

    subLeftImg = ((matrix[T:,:] - matrix[:-T,:]) - diffThreshold).clip(min=0,max=1)
    subRightImg = ((matrix[:-T,:] - matrix[T:,:]) - diffThreshold).clip(min=0,max=1)
    subHImg = np.bitwise_and(subLeftImg[:-T,:], subRightImg[T:,:])

    subTopImg = ((matrix[:,T:] - matrix[:,:-T]) - diffThreshold).clip(min=0,max=1)
    subBottomImg = ((matrix[:,:-T] - matrix[:,T:]) - diffThreshold).clip(min=0,max=1)
    subVImg = np.bitwise_and(subTopImg[:,:-T], subBottomImg[:,T:])

    ret = np.bitwise_or(subHImg[:,T:-T], subVImg[T:-T,:])
    ret = np.bitwise_and(lowest[T:-T,T:-T], ret)

    ret = np.pad(ret, T)
    return np.array(ret * 255, np.uint8)

  def calcM(self, modelCrossPoint, crossPoint):
    # M,_ = cv2.findHomography(sp.reshape((1,-1,3)), crossPoint.reshape((1,-1,2)))
    M,_ = cv2.findHomography(np.float32(modelCrossPoint), np.float32(crossPoint))
    return M
    
  def calcStandardLines(self, modelCrossPoint, crossPoint):
      M = self.calcM(modelCrossPoint, crossPoint)
      # print("M = ", M)
      srcLines = np.array(np.reshape(self.standardLines, (len(self.standardLines)*2,2)), dtype='float32')

      sLines = cv2.perspectiveTransform(np.asarray([srcLines]), M)
      sLines = np.array(np.reshape(sLines[0], (int(len(sLines[0])/2),4)), dtype="int")

      self.M = np.linalg.inv(M)
      return sLines

  def scoreCalc(self, modelCrossPoint, crossPoint):
    sLines = self.calcStandardLines(modelCrossPoint, crossPoint) 

    rect = cv2.boundingRect(np.array(sLines).reshape((-1,2)))
    # area = cv2.contourArea(np.array(sLines).reshape((-1,2)))
    area = rect[2] * rect[3]
    imgArea = self.img.shape[0] * self.img.shape[1]
    if area < imgArea / 2 or area > imgArea * 1.2 :
      return 0

    lineImg = np.zeros((self.img_binary.shape), np.uint8)
    # for l in sLines[0:2]:
    #   cv2.line(lineImg, l[0:2], l[2:4], 255, 4)
    # for l in sLines[12:14]:
    #   cv2.line(lineImg, l[0:2], l[2:4], 255, 4)
    for l in sLines:
      cv2.line(lineImg, l[0:2], l[2:4], 255, 4)
    
    crossImg = cv2.bitwise_and(self.img_binary, self.img_binary, mask = lineImg)
    score = len(np.where(crossImg == 255)[0])

    # if score > 10000:
    #   print("area: ", area)
    return score

  def fitModel(self, preMatchLines, modelLines, imgLines):
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
    iLeftLines, iRightLines = list(imgLines)

    lLength = len(mLeftLines)
    rLength = len(mRightLines)
    
    # 对左右两边的线条进行组合
    if lLength > 1:
      leftLinesCom = list(itertools.combinations(range(len(iLeftLines)), lLength))
      leftLinesCom = sorted(leftLinesCom, key = lambda i: max(i))
      leftLinesCom = [[iLeftLines[j] for j in i] for i in leftLinesCom]
    else:
      leftLinesCom = [[i] for i in iLeftLines]

    if rLength > 1:
      rightLinesCom = list(itertools.combinations(range(len(iRightLines)), rLength))
      rightLinesCom = sorted(rightLinesCom, key = lambda i: max(i))
      rightLinesCom = [[iRightLines[j] for j in i] for i in rightLinesCom]
    else:
      rightLinesCom = [[i] for i in iRightLines]

    # 从小到大的组合两个数组，目的是优先检测低索引的组合
    lineSetIdx = itertools.product(range(len(leftLinesCom)), range(len(rightLinesCom)))
    lineSetIdx = sorted(lineSetIdx, key = lambda i: max(i))
    lineSet = [(leftLinesCom[i[0]], rightLinesCom[i[1]]) for i in lineSetIdx]

    # 获取imglines交点
    imgCrossSet = list(map(lambda i:
      [utils.line_intersection(j[0], j[1]) for j in itertools.product(list(pLeftLines)+i[0], list(pRightLines)+i[1])], lineSet
    ))
    # 获取modellines交点
    modelCross = [utils.line_intersection(j[0], j[1]) for j in itertools.product(list(pmLeftLines) + mLeftLines, list(pmRightLines) + mRightLines)]
    

    # 计算分数
    # scores = [(idx, scoreCalc(modelCross, i)) for idx,i in enumerate(imgCrossSet)]
    scores = []
    best = 0
    limit = 0
    LIMIT_MAX = 150 # 找到 best 之后，max 次搜索后停止
    print("imgCrossSet size = ", len(imgCrossSet))
    for idx,i in enumerate(imgCrossSet):
      score = self.scoreCalc(modelCross, i)
      scores.append((idx, score))
      if score > best:
        limit = LIMIT_MAX
        best = score
        if self.DEBUG:
          calcedStardardLines = self.calcStandardLines(modelCross, imgCrossSet[int(idx)])

          rect = cv2.boundingRect(np.array(calcedStardardLines).reshape((-1,2)))
          # area = cv2.contourArea(np.array(calcedStardardLines).reshape((-1,2)))
          area = rect[2] * rect[3]
          print("found best: ", idx, " score: ", best, " area: ", area)
          # print("found score: ", best)
          # print("found area: ", area)
          utils.printLine(self.img, list(np.reshape(lineSet[idx],(-1,4))) + list(calcedStardardLines) , "test2-best.jpg")
          # time.sleep(1)

      if best > 0 and limit < 0: break
      limit -= 1
      
    scores = sorted(scores, key=lambda i: -i[1])
    targetIdx = int(scores[0][0])
    calcedStardardLines = self.calcStandardLines(modelCross, imgCrossSet[targetIdx])

    targetLineSetIdx = lineSetIdx[targetIdx]
    targetLeftLinesCom = leftLinesCom[targetLineSetIdx[0]]
    targetRightLinesCom = rightLinesCom[targetLineSetIdx[1]]

    lIdx = np.where(self.leftLines == targetLeftLinesCom[-1])[0][0]
    rIdx = np.where(self.rightLines == targetRightLinesCom[-1])[0][0]

    return calcedStardardLines, lineSet[targetIdx], lIdx, rIdx

  def find(self, img):

    TS_BEGIN = time.time()
    print("Step0 start")
    self.img = img

    ###
    ### 1. 取亮度，使用线条过滤器进行二值化，使用小物体过滤器去除小物体干扰
    ###
    TS = time.time()
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    self.img_lum = np.zeros(self.img.shape[:2], dtype=np.uint8)
    self.img_lum[:,:] = ycrcb[:,:,0]

    # 中值滤波，去噪
    self.img_lum = cv2.medianBlur(self.img_lum, 3)

    self.DEBUG and cv2.imwrite("test2-lum.jpg", cv2.cvtColor(self.img_lum, cv2.COLOR_GRAY2BGR))

    # 比较粗的线
    self.img_binary = self.getLinePixel(self.img_lum, max(self.img_lum.shape[:2])/60)
    # 比较细的线
    binary_ex = self.getLinePixel(self.img_lum, max(self.img_lum.shape[:2])/600)
    self.img_binary = cv2.bitwise_or(self.img_binary, binary_ex)

    # 删除那些很不像白色的像素
    cr = np.array(ycrcb[:,:,1], np.int)
    cb = np.array(ycrcb[:,:,2], np.int)
    white = cv2.inRange(cr - cb, -30, 30)
    self.img_binary = cv2.bitwise_and(self.img_binary, self.img_binary, mask=white)

    self.DEBUG and cv2.imwrite("test2-binary.jpg", self.img_binary)
    
    self.img_binary = self.removeSmallObject(self.img_binary)

    self.DEBUG and cv2.imwrite("test2-binary-removeSmallObject.jpg", self.img_binary)

    print("Step1 time: ", time.time() - TS)

    ###
    ### 2. 边缘化，分析出边线
    ###

    TS = time.time()

    self.img_canny = cv2.Canny(self.img_binary, threshold1=80, threshold2=200, apertureSize=7)
    self.DEBUG and cv2.imwrite("test2-canny.jpg", self.img_canny)


    self.lines = cv2.HoughLinesP(self.img_canny, 
      rho=1, 
      theta=np.pi / 180, 
      threshold=50, 
      minLineLength=50, 
      maxLineGap=50)
    self.lines = [i[0] for i in self.lines]
    self.DEBUG and utils.printLine(self.img, self.lines, "test2-lines.jpg")

    self.lines = [self.refineLine(i, self.img_canny) for i in self.lines]
    self.DEBUG and utils.printLine(self.img, self.lines, "test2-lines-refined.jpg")

    self.lines = self.merge_lines2(self.lines)
    self.DEBUG and utils.printLine(self.img, self.lines, "test2-lines-merged.jpg")

    print("Step2 time: ", time.time() - TS)

    ###
    ### 3. 根据线条角度分出横竖两组，并排序
    ###

    TS = time.time()

    leftLines = []
    rightLines = []

    for line in self.lines:
        deg = utils.get_orientation(line)
        if deg > 0:
            leftLines.append(line)
        else:
            rightLines.append(line)

    leftLines = sorted(leftLines, key=lambda i: 
        utils.point_to_line_distance3([0, img.shape[0]],i))
    rightLines = sorted(rightLines, key=lambda i: 
        utils.point_to_line_distance3([img.shape[1], img.shape[0]],i))

    # leftLines = leftLines[0:35]
    # rightLines = rightLines[0:35]

    self.DEBUG and utils.printLine(self.img, leftLines, "test2-lines-left.jpg")
    self.DEBUG and utils.printLine(self.img, rightLines, "test2-lines-right.jpg")

    self.leftLines = leftLines
    self.rightLines = rightLines

    print("Step3 time: ", time.time() - TS)

    ###
    ### 4. 开始寻找映射矩阵
    ###

    TS = time.time()

    ModelLines = [
      [
      # 左上角矩形
          [
              self.standardLines[-10],
              self.standardLines[-8],
          ],
          [
              self.standardLines[0], 
              self.standardLines[2],
          ]
      ],[
      # 左上角最大的矩形
          [
              self.standardLines[-6],
          ],
          [
              self.standardLines[4],
          ]
      ],[
      # 全场
          [
              self.standardLines[-4],
          ],
          [
              self.standardLines[6],
          ]
      # ],[
      # # 全场
      #     [
      #         self.standardLines[-1],
      #     ],
      #     [
      #         self.standardLines[11],
      #     ]
      ]
    ]

    matchs = [[[],[]],[[],[]]]
    lIdx = rIdx = -1
    for i in range(len(ModelLines)):
      calcedStardardLines, matchLines, lIdx, rIdx = self.fitModel(
        matchs, ModelLines[i], [leftLines[lIdx+1:], rightLines[rIdx+1:]])

      matchs[0][0] += ModelLines[i][0]
      matchs[0][1] += ModelLines[i][1]
      matchs[1][0] += matchLines[0]
      matchs[1][1] += matchLines[1]

    self.DEBUG and utils.printLine(self.img, calcedStardardLines, "test2-standard.jpg")

    print("Step4 time: ", time.time() - TS)
    print("Total time: ", time.time() - TS_BEGIN)
