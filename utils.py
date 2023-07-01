"""
! author: enpei
! Date: 2021-12-23
封装常用工具，降低Demo复杂度
"""
# 导入PIL
from PIL import Image, ImageDraw, ImageFont
# 导入OpenCV
import cv2
from matplotlib.pyplot import xlabel
import numpy as np
import time
import os
import glob
import math
from sys import platform as _platform 


class Utils:
    def __init__(self):
        self.disx=0
        self.disy=0 
        self.is_eye_watch='normal' 
        pass
    def trace(self,frame,faces,face_detector,width,height,landmarks): 
        # left_x = int(image_points[36][0])
        # left_y = int(image_points[36][1])
        # right_x = int(image_points[45][0])
        # right_y = int(image_points[45][1])
        
        x = landmarks.part(36).x
        y = landmarks.part(36).y
        print(x,y)
        yl=y-25
        yr=y+50
        xl=x-25
        xr=x+50
        roi = frame[yl: yr, xl: xr]  #利用切片工具，选出感兴趣roi区域

        rows, cols, _ = roi.shape   #保存视频尺寸以备用
        frows, fcols, _ = frame.shape
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)   #转灰度
        gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)    #高斯滤波一次
        
        _, threshold = cv2.threshold(gray_roi, 80, 255, cv2.THRESH_BINARY_INV)  #二值化，依据需要改变阈值
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #画连通域
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) 
        is_eye_watch='normal'
        f=0
        for cnt in contours:
            f=1
            (x, y, w, h) = cv2.boundingRect(cnt)
            x=x+xl
            y=y+yl#(x,y)是瞳孔的坐标
            
            if self.Judge(x,y,faces[0])==1: 
                frame = self.cv2AddChineseText(frame, "乱瞟: True ",  (20, 280), textColor=(178, 34 ,34), textSize=50) 
                # print(777777777777777)  
                is_eye_watch='warning' 
            else:
                frame = self.cv2AddChineseText(frame, "乱瞟: false ",  (20, 280), textColor=(0, 255, 0), textSize=50)
                is_eye_watch='normal'    
            cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), frows), (0, 255, 0), 2)
            cv2.line(frame, (0, y + int(h/2)), (fcols, y + int(h/2)), (0, 255, 0), 2)
            break
        if f==0:
            frame = self.cv2AddChineseText(frame, "乱瞟: True ",  (20, 280), textColor=(178, 34 ,34), textSize=50) 
            is_eye_watch='warning' 
        cv2.imshow("Roi", roi)
        cv2.imshow("Threshold", threshold)
        key = cv2.waitKey(5)
        return is_eye_watch,frame 
        #
        #
        #
    def Judge(self,x,y,face):
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        print(self.disx,x-x1)
        if(self.disx==0):
            self.disx=x-x1
        else:
            if(math.fabs(x-x1-self.disx)>4):
                return 1
            else:
                return 0
    # 添加中文
    def cv2AddChineseText(self,img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "./fonts/simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def getFaceXY(self,face):
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        return x1,y1,x2,y2

    # 人脸框框
    def draw_face_box(self,face,frame,zh_name,is_watch,distance):

        color = (255, 0, 255) if is_watch =='是' else (0, 255, 0)
        x1,y1,x2,y2 = self.getFaceXY(face)
        
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        m_str = '' if distance =='' else 'm'
        frame = self.cv2AddChineseText(frame, "{name} {distance}{m_str}".format(name = zh_name,distance=distance,m_str=m_str),  (x1, y1-60), textColor=color, textSize=50)

        return frame
    
    
    
   # 保存人脸照片
    def save_face(self,face,frame,face_name,password):

        path = './face_imgs/'+str(face_name)
        if not os.path.exists(path):
            os.makedirs(path)
        x1,y1,x2,y2 = self.getFaceXY(face)
        face_img = frame[y1:y2,x1:x2]

        filename = path+'/'+str(face_name)+'-'+str(time.time())+'.jpg'
        
        cv2.imwrite(filename,face_img) 

    # 人脸点
    def draw_face_points(self,landmarks,frame):
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 255), -1)
    
    # 获取人脸和label数据
    def getFacesLabels(self):
        # 遍历所有文件
        label_list = []
        img_list = []
        for file_path in glob.glob('./face_imgs/*/*'):

            dir_str = '/'         
            if _platform == "linux" or _platform == "linux2":
                # linux
                pass
            elif _platform == "darwin":
                # MAC OS X
                pass
            elif _platform == "win32":
                # Windows
                dir_str = '\\'
            elif _platform == "win64":
                # Windows 64-bit
                dir_str = '\\'

            label_list.append(int( file_path.split(dir_str)[-1].split('-')[0] ))


            img = cv2.imread(file_path)
            # img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_list.append(img)

        return np.array(label_list),img_list
     # 获取人脸和label数据
    def getFacesFileName(self,name,password): 
        # 遍历所有文件
        label_list = []
        img_list = []
        for file_path in glob.glob('.\\face_imgs\\*'):
            print(file_path)
            print(file_path.split('\\')[-1])
            if(file_path.split('\\')[-1]==name):
                img_pathlist=[]
                for img_path in glob.glob('.\\face_imgs\\'+name+'\\*'):
                    print(img_path)
                    img_pathlist.append(img_path)
                return img_pathlist
        return "NotFound"
    # 加载label 对应中文
    def loadLablZh(self):
        with open('./face_model/label.txt',encoding='utf-8') as f:

            back_dict = {}
            for line in f.readlines():
                label_index = line.split(',')[0]
                label_zh = line.split(',')[1].split('\n')[0]
                back_dict[label_index] = label_zh


            return back_dict