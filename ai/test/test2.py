import cv2
import numpy as np
import os
import math

from test3 import HoughBundler

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

# 读入图片
img = cv2.imread("b.jpg")

# https://blog.csdn.net/weixin_44120025/article/details/122102011
# K = np.zeros((3, 3))
# K[0, 0] = 1201.58
# K[1, 1] = 1201.72
# K[0, 2] = 1019.59
# K[1, 2] = 807.568
# K[2, 2] = 1

# distCoeffs = np.float32([-0.0911113, 0.0852054, 1.79509e-06, 0.000242446])

# img = cv2.undistort(img, K, distCoeffs)
# cv2.imwrite("test2-distort.jpg", img)

# 中值滤波，去噪
img = cv2.medianBlur(img, 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('original', gray)
# 阈值分割得到二值化图片
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ret, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY )
# binary = np.invert(binary)
flood_fill(binary, 0, 0, 0, 2)
cv2.imwrite("test2-a.jpg", binary)
# 膨胀操作
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_clo = cv2.dilate(binary, kernel2, iterations=2)
# 连通域分析
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)
# 查看各个返回值
# 连通域数量
print('num_labels = ',num_labels)
# 连通域的信息：对应各个轮廓的x、y、width、height和面积
print('stats = ',stats)
# 连通域的中心点
print('centroids = ',centroids)
# 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
print('labels = ',labels)
# 不同的连通域赋予不同的颜色
output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
for i in range(1, num_labels):
    mask = labels == i
    output[:, :, 0][mask] = np.random.randint(0, 255)
    output[:, :, 1][mask] = np.random.randint(0, 255)
    output[:, :, 2][mask] = np.random.randint(0, 255)
cv2.imwrite("test2.jpg", output)

# edges = cv2.Canny(binary, 55, 110)
edges = cv2.Canny(binary, threshold1=50, threshold2=200, apertureSize=7)
cv2.imwrite("test2-edges.jpg", edges)

#HoughLinesP方法判断哪些边缘是直线
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments
# lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#                     min_line_length, max_line_gap)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

# lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8) 
line_color = [0, 255, 0]
line_thickness = 2
dot_color = [0, 255, 0]
dot_size = 3

# dec = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
# lines = dec.detect(binary)

print('lines = ', lines.shape[0])
print('lines = ', lines)

# for r_theta in lines:
#     arr = np.array(r_theta[0], dtype=np.float64)
#     r, theta = arr
#     # Stores the value of cos(theta) in a
#     a = np.cos(theta)
 
#     # Stores the value of sin(theta) in b
#     b = np.sin(theta)
 
#     # x0 stores the value rcos(theta)
#     x0 = a*r
 
#     # y0 stores the value rsin(theta)
#     y0 = b*r
 
#     # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
#     x1 = int(x0 + 1000*(-b))
 
#     # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
#     y1 = int(y0 + 1000*(a))
 
#     # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
#     x2 = int(x0 - 1000*(-b))
 
#     # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)



# cv2.imwrite("test2-final.jpg", img)

#讲检测的直线叠加到原图
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_img, (x1, y1), (x2, y2), line_color, line_thickness)
        cv2.circle(line_img, (x1, y1), dot_size, dot_color, -1)
        cv2.circle(line_img, (x2, y2), dot_size, dot_color, -1)
final = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

cv2.imwrite("test2-lines.jpg", final)

def get_orientation( line):
    orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
    return math.degrees(orientation)
def extend_line_h0(line, height=img.shape[0]):
    # print('extend: ', line)
    deg = get_orientation(line[0])
    if not (45 < deg <= 90):
        return line
    x1, y1, x2, y2 = line[0]
    topRate = (y2 - y1) / (y1 - 0)
    bottomRate = (y2 - y1) / (height - y2)
    if topRate > 0:
        nx1 = x1 - (x2 - x1) / topRate 
    else:
        nx1 = x1
    if bottomRate > 0:
        nx2 = x2 + (x2 - x1) / bottomRate 
    else:
        nx2 = x2

    return [[nx1, 0, nx2, height]]

def extend_line_w0(line, width=img.shape[1]):
    # print('extend: ', line)
    deg = get_orientation(line[0])
    if (45 < deg <= 90):
        return line
    x1, y1, x2, y2 = line[0]
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

    return [[0, ny1, width, ny2]]
# Merge
lines = np.int16(list(map(extend_line_w0, lines)))
print('lines extended = ', lines)

bundler = HoughBundler(min_distance=5,min_angle=1)
lines = bundler.process_lines(lines)

print('lines merged = ', lines.shape[0])

line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8) 
#讲检测的直线叠加到原图
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_img, (x1, y1), (x2, y2), line_color, line_thickness)
        cv2.circle(line_img, (x1, y1), dot_size, dot_color, -1)
        cv2.circle(line_img, (x2, y2), dot_size, dot_color, -1)
final = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

cv2.imwrite("test2-merge.jpg", final)

# cv2.imshow('oginal', output)
# cv2.waitKey()
# cv2.destroyAllWindows()