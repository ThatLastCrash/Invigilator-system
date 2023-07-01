# 基于眼球追踪以及人脸姿态估计的监考系统

声明：本项目基于https://github.com/enpeizhao/CVprojects进行了一定的修改得到

#### 一、功能：

* 人脸识别：检查谁在看
* 眼球跟踪：检查是否眼睛乱瞟
* 人脸姿态估计：检查是否在看

#### 二、硬件：

* Windows10或11（无需GPU）或MacOS 都测试可行
* 普通RBG USB摄像头

#### 三、软件：

* python 3.7.10

`pip`安装一下依赖包

```
dlib
opencv-contrib-python（可能需要先卸载opencv-python：pip uninstall opencv-python）
```

下载权重文件(https://github.com/enpeizhao/CVprojects/releases/tag/Models)`shape_predictor_68_face_landmarks.dat`，放入`./assets`目录。

#### 四、使用方法：

运行demo.py后依据可视化界面操作

1.新用户需要注册，输入学号姓名后点击注册，几秒内会调用摄像头抓拍人脸，拍摄三张后人脸信息采集完成

2.注册完成后输入学号姓名

3.下面介绍六个模式

（1）只显示人脸框
（2）画出人脸68个关键点
（3）画出人脸梯形方向
（4）画出人脸朝向指针
（5）画出人脸的三维坐标系
（6）依据眼球跟踪及人脸姿态估计判定是否给出warning（主要模式）

