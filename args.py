# -*- coding: utf-8 -*-
"""
Created on Tue May 24 08:21:50 2022

@author: PC
"""

class Args:
    def __init__(self,mode='distance',label_id=1,img_count=3,img_interval=1,display=2,w=960,h=720):   
        self.mode=mode
        self.label_id=label_id  
        self.img_count=img_count
        self.img_interval=img_interval
        self.display=display
        self.w=w
        self.h=h  
    def set_mode(self,mode):
        self.mode=mode 
    '''
        parser = ArgumentParser()
parser.add_argument("--mode", type=str, default='run',     
                    help="运行模式：collect,train,distance,run对应：采集、训练、评估距离、主程序")
parser.add_argument("--label_id", type=int, default=1,
                    help="采集照片标签id.")
parser.add_argument("--img_count", type=int, default=3,
                    help="采集照片数量")        
parser.add_argument("--img_interval", type=int, default=3, 
                    help="采集照片间隔时间")            
                    
parser.add_argument("--display", type=int, default=1,
                    help="显示模式，取值1-5")     
                     
parser.add_argument("--w", type=int, default=960,
                    help="画面宽度")   
parser.add_argument("--h", type=int, default=720,
                    help="画面高度")  
'''