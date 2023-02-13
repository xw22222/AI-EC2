import boto3
import os
import torch
from PIL import Image, ImageFilter
import easyocr
from cloudpathlib import CloudPath
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import PIL
import numpy as np
import linecache
import pandas as pd
import io
from operator import itemgetter

S3 = boto3.client('s3')
bucket = '1iotjj'
path = './crops'
labels_Path = './labels'
input_path = './input_img/'

# 가장 최근 생성된(생성시각 이 큰) 파일을 리턴하는 함수 
def recently(folder_path) :
    each_file_path_and_gen_time = []
    for each_file_name in os.listdir(folder_path):
    # getctime: 입력받은 경로에 대한 생성 시간을 리턴
        each_file_path = folder_path + each_file_name
        each_file_gen_time = os.path.getctime(each_file_path)
        each_file_path_and_gen_time.append(
        (each_file_path, each_file_gen_time)
    )
    return max(each_file_path_and_gen_time, key=lambda x: x[1])[0]

def YOLO(path) :
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./V2.pt', force_reload=True)
    # 가장 최근 생성된 차량 이미지 읽기 
    img = Image.open(recently(path)) # PIL
    results = model(img, size=640) # 이미지 크롭
    df = results.pandas().xyxy[0]
    crops = results.crop(save=False) # ./test_crops1 dir 생성 요
  #  conf = (crop[0]['conf'].item() * 100)
    for crop in enumerate(crops) :
        print(crop.item())

"""
        if 'plate' in crop['label'] and crop['conf'].item() * 100 > 0:
            # if 'plate' in crop['label'] and crop['conf'].item() 
            image = crop['im']
            im = Image.fromarray(image)   
            im.save(os.path.join('./crops2' , f'plate_result.png'), 'png',dpi=(300,300))
    
    torch.save(results, '/labels')
    loaded_model = torch.load('./labels')
    for p in loaded_model.parameters():
        f = open(f'labels.txt', 'w')
        f.write(p)
        f.close
        """
"""


#python detect.py --weights runs/train/exp2/weights/best.pt --img 640 --conf 0.1 --source Tayo2-3/test/images \--save-conf \--save-txt
#이놈이 이미지를 가져와서 돌린 결과의 labels의 경로를 잡고 그경롤 YOLO_2에 넣어야됨 

def YOLO_2 (path) :
    input = recently(input_path) # 가장 최근 수신된 이미지를 받는 변수 input
    resultname = os.path.basename(input)    # 파일에서 이름만 resultname으로 가져옴/ 버킷에 올릴때만 필요 
    li = []
    with open(path) as f:
      lines = f.readlines()
      for line in lines:
        sp = line.split()
        if float(sp[5]) < 0.4:
          pass
        else:
          li.append(sp)
    li = sorted(li, key=itemgetter(1))
    result = ''.join(s[0] for s in li)
    print(path)
    print(result[-4:])
    print(result[-4:])

    print(f"결과 save & S3 업로드 : {result}.txt ")
    print("AWS S3 Upload path : 1iotjj/carnum2")
    f = open(f'carnum2.txt','w')    # carnum2.txt 파일 생성 /.(루트)
    f.write(result) #결과부분 만 쓰기 모드로 작성
    f.close()
    S3.upload_file(f'carnum2.txt', bucket,'carnum2/'+ f'{resultname}.txt') #S3/carnum dir에 최근 input_img.txt로 업로드

for path in glob.glob('/content/drive/MyDrive/unite/yolov5/runs/detect/exp5/labels/*.txt'):
  li = []
  with open(path) as f:
    lines = f.readlines()
    for line in lines:
      sp = line.split()
      if float(sp[5]) < 0.4:
        # print(str(path)[60:68]+' has bad detection')
        pass
      else:
        li.append(sp)

  li = sorted(li, key=itemgetter(1))
  result = ''.join(s[0] for s in li)

  print(path)
  print(result[-4:])

"""

YOLO(input_path)