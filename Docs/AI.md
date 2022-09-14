# 通过卷积网络自动判断得分
功能
- 落球点标记，界内，界外，滚网
- 标记每个人得分/失分
- 标记每个人助攻
- 技术动作统计（杀球，平抽，吊球）
- 球速记录
## 需要的技术
- 场地识别
- 羽毛球追踪
- 人物追踪
- 三维重建

## note
- https://xugaoxiang.com/2020/10/17/yolov5-deepsort-pytorch/
git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git
YOLOv5 + StrongSORT
人物监测很健壮，人物跟踪偶尔会失败，例如遮挡太久。
球拍跟踪基本不行，应该是训练集的问题
羽毛球无法跟踪，应该没有相关训练集
T4
yolo(20ms) + strongSORT(50ms) / frame

- https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2
羽毛球跟踪位置抖动比较厉害，比较容易出现检测错误，应该与训练集太小有关系。
T4
10fps

- https://github.com/PaddlePaddle/PaddleDetection
性能优于 yolov5 的目标检测，跟踪
里面有使用 JDE(Joint Detection and Embedding) 检测与跟踪一体化算法（FairMOT），可以做到实时检测跟踪

- https://github.com/litinglin/swintrack
一个性能优异的目标跟踪器

- https://github.com/PaddlePaddle/PaddleSpeech#SpeechToText
百度大脑的飞桨包含一个语音识别开源项目

- 两份羽毛球跟踪的论文
https://www.docin.com/p-2278980712.html
https://www.doc88.com/p-9913921536756.html

- opencv 通过两幅图片，找到他们之间的三维变换矩阵
  https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html#epipolar-geometry
  https://stackoverflow.com/questions/10744671/reconstruction-a-3d-point-from-two-2d-points

- 带阻力抛物线
  - https://www.bilibili.com/read/cv9152884/
- 解常微分方程
  - https://blog.csdn.net/ouening/article/details/80673288
  - https://vlight.me/2018/05/01/Numerical-Python-Ordinary-Differential-Equations/
  - 