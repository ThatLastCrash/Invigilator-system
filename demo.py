"""
! author: enpei
! date: 2021-12-23
主要功能：检测孩子是否在看电视，看了多久，距离多远
使用技术点：人脸检测、人脸识别（采集照片、训练、识别）、姿态估计
""" 
import cv2,time
from pose_estimator import PoseEstimator   
import dlib
import tkinter as tk 
from tkinter import messagebox 
from utils import Utils
from args import Args 
import os
import numpy as np 
import pickle  
import matplotlib.pyplot as plt  
import math 
import face_recognition 
class MonitorBabay:
    def __init__(self):
        # 人脸检测
        self.face_detector = dlib.get_frontal_face_detector()
        # 人脸识别模型：pip uninstall opencv-python，pip install opencv-contrib-python
        self.face_model = cv2.face.LBPHFaceRecognizer_create()

        # 人脸68个关键点
        self.landmark_predictor = dlib.shape_predictor("./assets/shape_predictor_68_face_landmarks.dat")

        # 站在1.5M远处，左眼最左边距离右眼最右边的像素距离(请使用getEyePixelDist方法校准，然后修改这里的值)
        self.eyeBaseDistance = 40  

        # pose_estimator.show_3d_model()

        self.utils = Utils()


    # 采集照片用于训练
    # 参数
    # label_index: label的索引
    # save_interval：隔几秒存储照片
    # save_num：存储总量
    def collectFacesFromCamera(self,name,save_interval,password): 
        cap = cv2.VideoCapture(0)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

        fpsTime = time.time()
        last_save_time = fpsTime
        saved_num = 0
        save_num=3
        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector(gray)
            for face in faces:

                if saved_num < save_num:
                    if (time.time() - last_save_time) > save_interval:
                        self.utils.save_face(face,frame,name,password)
                        saved_num +=1
                        last_save_time = time.time()

                        print('name:{index}，成功采集第{num}张照片'.format(index = name,num = saved_num))
                else:
                    print('照片采集完毕！')
                    #exit()
                    return

                self.utils.draw_face_box(face,frame,'','','')    

            cTime = time.time()
            fps_text = 1/(cTime-fpsTime)
            fpsTime = cTime
            
            frame = self.utils.cv2AddChineseText(frame, "帧率: " + str(int(fps_text)),  (10, 30), textColor=(0, 255, 0), textSize=50)
            frame = cv2.resize(frame, (int(width)//2, int(height)//2) )
            cv2.imshow('Collect data', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


    # 训练人脸模型 
    def train(self):
        print('训练开始！')
        label_list,img_list = self.utils.getFacesLabels()
        self.face_model.train(img_list, label_list)
        self.face_model.save("./face_model/model.yml")
        print('训练完毕！')
    

    
    # 获取两个眼角像素距离
    def getEyePixelDist(self):
        
        #cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture("obs录屏有眼镜.mp4")

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 姿态估计
        self.pose_estimator = PoseEstimator(img_size=(height, width))
        
        fpsTime = time.time()

        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector(gray)
           
            pixel_dist = 0

            for face in faces:
                
                # 关键点
                landmarks = self.landmark_predictor(gray, face)
                image_points = self.pose_estimator.get_image_points(landmarks)

                left_x = int(image_points[36][0])
                left_y = int(image_points[36][1])
                right_x = int(image_points[45][0])
                right_y = int(image_points[45][1])

                pixel_dist = abs(right_x-left_x)

                cv2.circle(frame, (left_x, left_y), 8, (255, 0, 255), -1)
                cv2.circle(frame, (right_x, right_y), 8, (255, 0, 255), -1)

                # 人脸框
                frame = self.utils.draw_face_box(face,frame,'','','')
              

            cTime = time.time()
            fps_text = 1/(cTime-fpsTime)
            fpsTime = cTime
            
            frame = self.utils.cv2AddChineseText(frame, "帧率: " + str(int(fps_text)),  (20, 30), textColor=(0, 255, 0), textSize=50)
            frame = self.utils.cv2AddChineseText(frame, "像素距离: " + str(int(pixel_dist)),  (20, 100), textColor=(0, 255, 0), textSize=50)
           
            # frame = cv2.resize(frame, (int(width)//2, int(height)//2) )
            cv2.imshow('Baby wathching TV', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

    # 运行主程序
    def run(self,w,h,display,name,password):

        
        model_path = "./face_model/model.yml"
        if not os.path.exists(model_path):
            print('人脸识别模型文件不存在，请先采集训练')
            exit()

        label_zh = self.utils.loadLablZh()

        self.face_model.read(model_path)

        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("obs录屏有眼镜.mkv")    
        width = w
        height = h

        print(width,height)

        # 姿态估计
        self.pose_estimator = PoseEstimator(img_size=(height, width)) 
        fpsTime = time.time()

        zh_name = ''
        x_label = ''
        z_label = ''
        is_watch = ''
        angles = [0,0,0]
        person_distance = 00
        '''
        watch_start_time = fpsTime  
        watch_duration = 0
        '''
        # fps = 12
        # videoWriter = cv2.VideoWriter('./record_video/out'+str(time.time())+'.mp4', cv2.VideoWriter_fourcc(*'H264'), fps, (width,height))
        '''
        获取屏幕大小 
        '''
        
        stk = tk.Tk()
        screen_width = stk.winfo_screenmmwidth()
        screen_height = stk.winfo_screenmmheight() 
        print( screen_width/1.25+1.5) 
        print(screen_height/1.25+1.5)       
        stk.destroy()    
        #计算屏幕中心点 
        center_x=(screen_width/1.25+1.5)/2
        center_z=(screen_height/1.25+1.5)/2 
        
        #
        #
        #
        #face_recognition准备工作
        # Load a sample picture and learn how to recognize it.
        '''obama_image = face_recognition.load_image_file("jinjin.png")
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
        
        # Load a second sample picture and learn how to recognize it.
        biden_image = face_recognition.load_image_file("hu.png")
        biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
        
        # Create arrays of known face encodings and their names
        known_face_encodings = [
            obama_face_encoding,
            biden_face_encoding
        ]
        known_face_names = [
            "Jin",
            "Hu"
        ]'''
        
        # Load a sample picture and learn how to recognize it.
        image_filename=self.utils.getFacesFileName(name,password)
        
        for each_path in image_filename:
            if(each_path=="NotFound"):
                print(image_filename)
                print("未找到人脸照片")
            image = face_recognition.load_image_file(each_path)
            if len(face_recognition.face_encodings(image))>0:
                # 获取检测到人脸时面部编码信息中第一个面部编码
                face_encoding = face_recognition.face_encodings(image)[0]
                break
            else:
                print("未检测到有效人脸区域！")
        

        # Create arrays of known face encodings and their names
        known_face_encodings = [
            face_encoding
        ]
        known_face_names = [
            name
        ]
        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        #
        #
        #
        
        # fps = 12
        # videoWriter = cv2.VideoWriter('./record_video/out'+str(time.time())+'.mp4', cv2.VideoWriter_fourcc(*'H264'), fps, (width,height))

        cnt=0
        while True:
            cnt=cnt+1
            ret, frame = cap.read() 
            if ret is False:
                break
            if cnt%2==0:
                continue
            frame = cv2.resize(frame,(width,height))
            #frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

            faces = self.face_detector(gray)
            
            
            is_loc_watch="normal"
            is_eye_watch="normal"
            for face in faces:
                
                x1,y1,x2,y2 = self.utils.getFaceXY(face)  
                face_img = gray[y1:y2,x1:x2] 
                '''
                try:
                    # 人脸识别
                    idx, confidence = self.face_model.predict(face_img) 
                    print(idx)  
                    zh_name = label_zh[str(idx)]
                except cv2.error:
                    print('cv2.error') 
                ''' 
                # 关键点
                landmarks = self.landmark_predictor(gray, face)
                # 计算旋转矢量
                rotation_vector, translation_vector = self.pose_estimator.solve_pose_by_68_points(landmarks)

                # 计算距离
                person_distance = round(self.pose_estimator.get_distance(self.eyeBaseDistance),2)


                # 计算角度
                rmat, jac = cv2.Rodrigues(rotation_vector)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                
                x_label='normal'
                z_label='normal'  
                '''
                通过x，z角，以及距离来计算视线落点到屏幕中心的距离 
                '''
                '''
                if angles[1]<9 and angles[1]>-9 : 
                    x_label ='前'  
                if angles[0] < 2: 
                    z_label = "下"
                elif angles[0] > 7:    
                    z_label = "上"
                else:
                    z_label = "中"  
                '''
                #计算竖直高度
                dis_z=math.tan(abs(angles[0])*math.pi/180)*person_distance*1000   
                #计算水平距离
                dis_x=math.tan(abs(angles[1])*math.pi/180)*person_distance*1000   
                #计算落点
                loc_x=0
                loc_z=0
                if angles[1] >0:
                    loc_x=center_x+dis_x
                else:
                    loc_x=center_x-dis_x
                if angles[0]>0:
                    loc_z=center_z-dis_z
                else:
                    loc_z=center_z+dis_z 
                #print(loc_x)
                #print(loc_z) 
                #判断是否视线落在屏幕内
                is_loc_watch='normal' 
                is_eye_watch='normal' 
                is_watch = 'normal'  
                #超出屏幕范围则判定为警告
                if loc_x> screen_width/1.25+2.5 or loc_x<-0.5:
                    x_label='warning'
                    is_loc_watch='warning' 
                if loc_z <-0.5 or loc_z>screen_height/1.25+2.5 :
                    z_label='warning'
                    is_loc_watch='warning'  
                '''
                if is_watch == '是':
                    now = time.time()
                    watch_duration += ( now - watch_start_time)
                
                watch_start_time= time.time()
                ''' 
                if display == 1: 
                    # 人脸框
                    frame = self.utils.draw_face_box(face,frame,zh_name,is_watch,person_distance)
                if display == 2:
                    # 68个关键点
                    self.utils.draw_face_points(landmarks,frame)
                if display == 3: 
                    # 梯形方向
                    self.pose_estimator.draw_annotation_box(
                        frame, rotation_vector, translation_vector,is_watch)
                if display == 4:
                    # 指针
                    self.pose_estimator.draw_pointer(frame, rotation_vector, translation_vector)
                if display == 5:
                    # 三维坐标系
                    self.pose_estimator.draw_axes(frame, rotation_vector, translation_vector)
                if display == 6:
                    # 眼球跟踪 
                    is_eye_watch,frame=self.utils.trace(frame,faces,self.face_detector,width,height,landmarks)
                # if display==7:
                #     frame = self.utils.draw_face_box(face,frame,zh_name,is_watch,person_distance)
                #     frame=self.utils.trace(frame,faces,self.face_detector,width,height) 
                # 仅测试单人  
                break 
            # 
            #
            #
            #这部分是face_recognition的人脸识别
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
        
            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
        
                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]
        
                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
        
                    face_names.append(name)
        
            process_this_frame = not process_this_frame
        
        
            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
        
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            #
            #
            #
            #
            if is_loc_watch=='warning':
                is_watch='warning'  
            if is_loc_watch=='normal' and is_eye_watch=='warning':  
                print('---------------------------------------------------')    
                is_watch='warning'  
            '''
            print(is_loc_watch)  
            print(self.utils.is_eye_watch)
            print(is_watch)
            '''
            cTime = time.time()
            fps_text = 1/(cTime-fpsTime) 
            fpsTime = cTime
            
            frame = self.utils.cv2AddChineseText(frame, "帧率: " + str(int(fps_text)),  (20, 30), textColor=(0, 255, 0), textSize=50)
            
            color = (255, 0, 255) if person_distance <=1 else (0, 255, 0)

            frame = self.utils.cv2AddChineseText(frame, "距离: " + str(person_distance ) +"m",  (20, 100), textColor=color, textSize=50)

            color = (255, 0, 255) if is_watch =='normal' else (178, 34 ,34)

            frame = self.utils.cv2AddChineseText(frame, "状态: " + str(is_watch),  (20, 170), textColor=color, textSize=50)
            # 
            #duration_str = str(round((watch_duration/60),2)) +"min"

            #frame = self.utils.cv2AddChineseText(frame, "时长: " + duration_str, (20, 240), textColor= (0, 255, 0), textSize=50)



            color = (255, 0, 255) if x_label =='前' else (0, 255, 0) 
            
            frame = self.utils.cv2AddChineseText(frame, "X轴: {degree}° {x_label} ".format(x_label=str(x_label ),degree = str(int(angles[1]))) ,  (20, height-220), textColor=color, textSize=40)

            color = (255, 0, 255) if z_label =='中' else (0, 255, 0)

            frame = self.utils.cv2AddChineseText(frame, "Z轴: {degree}° {z_label}".format(z_label=str(z_label ),degree = str(int(angles[0]))) ,  (20, height-160), textColor=color, textSize=40)


            frame = self.utils.cv2AddChineseText(frame, "Y轴: {degree}°".format(degree = str(int(angles[2]) )),(20, height-100), textColor=(0, 255, 0), textSize=40)


            # videoWriter.write(frame) 
            # frame = cv2.resize(frame, (int(width)//2, int(height)//2) )
            cv2.imshow('Baby wathching TV', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

m = MonitorBabay()   
args=Args()  
print(args.display) 
mode = args.mode
  
'''
登陆窗口
'''
#窗口
window=tk.Tk()
window.title('登陆窗口') 
window.geometry('500x300') 
tk.Label(window,text='学号:').place(x=100,y=80)   
tk.Label(window,text='姓名:').place(x=100,y=120) 
tk.Label(window,text='模式:').place(x=100,y=160) 
#学号
var_usr_name=tk.StringVar()
e_name=tk.Entry(window,textvariable=var_usr_name)
e_name.place(x=160,y=80) 
#姓名 
var_usr_pwd=tk.StringVar()
e_pwd=tk.Entry(window,textvariable=var_usr_pwd) 
e_pwd.place(x=160,y=120)  
#模式
var_new_select=tk.StringVar() 
e_pwd=tk.Entry(window,textvariable=var_new_select) 
e_pwd.place(x=160,y=160)


#登录函数
def usr_log_in():
    #输入框获取用户名密码
    usr_name=var_usr_name.get()
    usr_pwd=var_usr_pwd.get() 
    usr_sel=var_new_select.get() 
    #从本地字典获取用户信息，如果没有则新建本地数据库
    try:
        with open('usr_info.pickle','rb') as usr_file: 
            usrs_info=pickle.load(usr_file)
    except FileNotFoundError:
        with open('usr_info.pickle','wb') as usr_file:
            usrs_info={'admin':'admin'}
            pickle.dump(usrs_info,usr_file)
    #判断用户名和密码是否匹配
    if usr_name in usrs_info:
        if usr_pwd == usrs_info[usr_name]:
            tk.messagebox.showinfo(title='welcome',
                                   message='欢迎您：'+usr_name)
            window.destroy()
            m.run(args.w,args.h,(int)(usr_sel),usr_name,usr_pwd)  
        else:
            tk.messagebox.showerror(message='姓名错误')
    #用户名密码不能为空
    elif usr_name=='' or usr_pwd=='' :
        tk.messagebox.showerror(message='学号或姓名为空')
    #不在数据库中弹出是否注册的框
    else:
        is_signup=tk.messagebox.askyesno('欢迎','您还没有注册，是否现在注册') 
        if is_signup:
            try:
                with open('usr_info.pickle','rb') as usr_file:
                    exist_usr_info=pickle.load(usr_file)
            except FileNotFoundError:
                exist_usr_info={}           
            
        #检查用户名存在、密码为空、密码前后不一致
            if  usr_name in exist_usr_info:
                tk.messagebox.showerror('错误','学号已存在')
            elif usr_pwd =='' or usr_name=='':
                    tk.messagebox.showerror('错误','学号或姓名为空')  
        #注册信息没有问题则将用户名密码写入数据库
            else:
                exist_usr_info[usr_name]=usr_pwd
                with open('usr_info.pickle','wb') as usr_file: 
                    pickle.dump(exist_usr_info,usr_file) 
                tk.messagebox.showinfo('欢迎','注册成功')
                '''
                print("即将采集照片.")
                if args.label_id and args.img_count and args.img_interval:
                    m.collectFacesFromCamera(args.label_id,args.img_interval,args.img_count)
                with open("./face_model/label.txt","a") as files:
                    files.writelines(str(args.label_id)+','+usr_name)        
                m.train()   
                '''
                m.collectFacesFromCamera( usr_name,2,usr_pwd)  
def usr_sign_up():
    #确认注册时的相应函数
    def signtowcg():
        #获取输入框内的内容
        nn=new_name.get()
        np=new_pwd.get()
        #本地加载已有用户信息,如果没有则已有用户信息为空
        try:
            with open('usr_info.pickle','rb') as usr_file:
                exist_usr_info=pickle.load(usr_file)
        except FileNotFoundError:
            exist_usr_info={}           
            
        #检查用户名存在、密码为空、密码前后不一致
        if nn in exist_usr_info:
            tk.messagebox.showerror('错误','学号已存在') 
        elif np =='' or nn=='':
            tk.messagebox.showerror('错误','学号或姓名为空')  
        #注册信息没有问题则将用户名密码写入数据库
        else:
            exist_usr_info[nn]=np
            with open('usr_info.pickle','wb') as usr_file:
                pickle.dump(exist_usr_info,usr_file)
            tk.messagebox.showinfo('欢迎','注册成功') 
            #注册成功关闭注册框
            window_sign_up.destroy()
            print("即将采集照片.")
            '''
            if args.label_id and args.img_count and args.img_interval:
                m.collectFacesFromCamera(args.label_id,args.img_interval,args.img_count)
            
            
            with open("./face_model/label.txt","a") as files: 
                files.writelines(str(args.label_id)+','+nn)          
            m.train()  
            '''
            m.collectFacesFromCamera(nn,2,np) 
    #新建注册界面
    window_sign_up=tk.Toplevel(window) 
    window_sign_up.geometry('350x200') 
    window_sign_up.title('注册')
    #用户名变量及标签、输入框
    new_name=tk.StringVar() 
    tk.Label(window_sign_up,text='学号：').place(x=10,y=10)
    tk.Entry(window_sign_up,textvariable=new_name).place(x=150,y=10)
    #密码变量及标签、输入框
    new_pwd=tk.StringVar() 
    tk.Label(window_sign_up,text='姓名：').place(x=10,y=50)  
    tk.Entry(window_sign_up,textvariable=new_pwd).place(x=150,y=50)         
    #确认注册按钮及位置
    bt_confirm_sign_up=tk.Button(window_sign_up,text='确认注册',
                                 command=signtowcg)
    bt_confirm_sign_up.place(x=150,y=130)
#退出的函数
def usr_sign_quit():
    window.destroy()
#登录 注册按钮
bt_login=tk.Button(window,text='登录',width=10,height=1,command=usr_log_in)
bt_login.place(x=140,y=230)
bt_logup=tk.Button(window,text='注册',width=10,height=1,command=usr_sign_up) 
bt_logup.place(x=280,y=230) 
'''
bt_logquit=tk.Button(window,text='退出',command=usr_sign_quit)
bt_logquit.place(x=280,y=230) 
''' 
#主循环
window.mainloop()
