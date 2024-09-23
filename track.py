from ultralytics import YOLO

import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath

# 加载官方或自定义模型
model = YOLO('best.pt')  # 加载一个官方的检测模型

# 使用模型进行追踪
results = model.track(source="/Users/jason/IdeaProjects/PeopleFlowDetection/yolov5-master/data/images/MOT16-02.mp4", show=True)  # 使用默认追踪器进行追踪
