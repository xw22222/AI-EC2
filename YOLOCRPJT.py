# AI 모델 코드 짠거 : 모니터링 삽질 많이한 코드 

import boto3
import os
import torch
from PIL import Image, ImageFilter
import easyocr
from cloudpathlib import CloudPath

S3 = boto3.client('s3')
bucket = '1iotjj'
path = './crops'
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

# OCR 결과 읽고 차량번호 저장해서 S3로 반환하는 함수
def easy_ocr (path) :
    input = recently(input_path) # 가장 최근 수신된 이미지를 받는 변수 input
    resultname = os.path.basename(input)   # 파일에서 이름만 resultname으로 가져옴/ 버킷에 올릴때만 필요 
    reader = easyocr.Reader(['ko'], gpu=True)
    result = reader.readtext(path)
    read_result = result[0][1]
    read_confid = int(round(result[0][2], 2) * 100)
    print("===== Crop Image OCR Read - Easy ======")
    print(f'Easy OCR 결과     : {read_result}') 
    print(f'Easy OCR 확률     : {read_confid}%')
    print(f"Easy ocr 결과 save : {resultname}.txt ")
    print("AWS S3 Upload path : 1iotjj/carnum")
    print("=======================================")
    f = open(f'carnum.txt','w')    # carnum.txt 파일 생성 /.(루트)
    f.write(read_result)   #OCR 결과부분만 쓰기 모드로 작성
    f.close()
    S3.upload_file(f'carnum.txt', bucket,'test_carnum/'+ f'{resultname}.txt') #S3/carnum dir에 최근 input_img.txt로 업로드


# --실행부 -- 
# yolo Model load : 타요타요 학습된 모델 경로 : 루트 dir에 : ./best.pt 따로 빼기
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', force_reload=True)
img = Image.open(recently(input_path)) # PIL # 가장 최근 생성된 차량 이미지 읽기
img = img.filter(ImageFilter.GaussianBlur(radius =1))
results = model(img, size=640) # 이미지 크롭 
df = results.pandas().xyxy[0]
crops = results.crop(save=False) # 크롭 결과 none save 
# conf = (crop[0]['conf'].item() * 100)
for num, crop in enumerate(crops) :
    if 'plate' in crop['label'] and crop['conf'].item() * 100 > 50 :
        image = crop['im']
        im = Image.fromarray(image)   
        im.save(os.path.join(path, f'plate_result.png'), 'png',dpi=(300,300))
        
    # 굳이 crop결과 쌓아서 유지 할 필요가 있는가?? -> 덮어쓰기
    # Plate_result.png : 차량이미지에서 번호판 부분만 추출된 이미지
#가장 최근 생성된 Crops 결과 이미지 easy_ocr 함수 읽기 
easy_ocr(recently('./crops/'))
