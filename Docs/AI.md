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
羽毛球跟踪位置抖动比较厉害，
T4
10fps