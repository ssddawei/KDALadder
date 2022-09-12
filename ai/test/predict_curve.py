import os
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
import csv
import sys

# os.chdir("/TrackNetv2/3_in_3_out")
try:
	input_video_path = sys.argv[1]
	input_csv_path = sys.argv[2]
	#output_video_path = sys.argv[3]
	if (not input_video_path) or (not input_csv_path):
		raise ''
except:
	print('usage: python3 show_trajectory.py <input_video_path> <input_csv_path>')
	exit(1)

with open(input_csv_path) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	frames = []
	x, y = [], []
	list1 = []
	for row in readCSV:
		list1.append(row)
	for i in range(1 , len(list1)):
		frames += [int(list1[i][0])]
		x += [int(float(list1[i][2]))]
		y += [int(float(list1[i][3]))]

output_video_path = input_video_path.split('.')[0] + "_trajectory.mp4"

q = queue.deque()
for i in range(0,20):
	q.appendleft(None)

#get video fps&video size
currentFrame= 0
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path,fourcc, fps, (output_width,output_height))

video.set(1,currentFrame); 
ret, img1 = video.read()
#write image to video
output_video.write(img1)
currentFrame +=1
#input must be float type
img1 = img1.astype(np.float32)

#capture frame-by-frame
video.set(1,currentFrame);
ret, img = video.read()
#write image to video
output_video.write(img)
currentFrame +=1
#input must be float type
img = img.astype(np.float32)

def calc(x, y):
	A = np.zeros((len(x), 3)) #构造一个范德蒙德矩阵
	A[:,0] = 1
	A[:,1] = x
	A[:,2] = x**2

	c = np.matmul(A.T, A)
	d = np.matmul(A.T, y)

	_,result = cv2.solve(c,d)
	return result

while(True):

	#capture frame-by-frame
	video.set(1,currentFrame); 
	ret, img = video.read()
		#if there dont have any frame in video, break
	if not ret: 
		break
	if currentFrame > 3000:
		break
	PIL_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
	PIL_image = Image.fromarray(PIL_image)
	if currentFrame >= len(x) or currentFrame >= len(y):
		break
	if x[currentFrame] != 0 and y[currentFrame] != 0:
		q.appendleft([x[currentFrame],y[currentFrame]])
		q.pop()
	else:
		q.appendleft(None)
		q.pop()

	calcX = []
	calcY = []
	for i in range(0,20):
		if q[i] is not None:
			draw_x = q[i][0]
			draw_y = q[i][1]
			if draw_x > 0 and draw_y > 0:
				calcX.append(draw_x)
				calcY.append(draw_y)
			bbox =  (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
			draw = ImageDraw.Draw(PIL_image)
			draw.ellipse(bbox, outline ='yellow')
			del draw
	if len(calcX):
		result = calc(np.array(calcX[:10]), np.array(calcY[:10]))
		
		x2 = np.array([output_width/100*i for i in range(1,100)])
		y2 = result[0] + result[1]*x2 + result[2] * (x2**2) 
		#print(calcX,calcY,x2,y2)
		for j in range(len(y2)):
			dx = x2[j]
			dy = y2[j]
			bbox =  (dx - 2, dy - 2, dx + 2, dy + 2)
			draw = ImageDraw.Draw(PIL_image)
			draw.ellipse(bbox, outline ='red')
			del draw

	
	opencvImage =  cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
	#write image to output_video
	output_video.write(opencvImage)

	#next frame
	currentFrame += 1

video.release()
output_video.release()
print("finish")

