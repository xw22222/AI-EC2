## YOLO 모델 V2 Run용 
import boto3
import os, sys, time
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from PIL import Image, ImageFilter
import easyocr
import pytesseract
import boto3

S3 = boto3.client('s3')
bucket = '1iotjj'
path = './crops'

# 가장 최근 생성된 파일을 리턴하는 함수 : 짠거
def recently(folder_path) :
    each_file_path_and_gen_time = []
    for each_file_name in os.listdir(folder_path):
    # getctime: 입력받은 경로에 대한 생성 시간을 리턴
        each_file_path = folder_path + each_file_name
        each_file_gen_time = os.path.getctime(each_file_path)
        each_file_path_and_gen_time.append(
        (each_file_path, each_file_gen_time)
    )
    # 가장 생성시각이 큰(가장 최근인) 파일을 리턴 
    return max(each_file_path_and_gen_time, key=lambda x: x[1])[0]


# OCR 결과 읽는 부분 차량번호만 .txt파일로 저장 예정 -> boto3 s3 
def easy_ocr (path) :
    reader = easyocr.Reader(['ko'], gpu=True)
    result = reader.readtext(path)
    read_result = result[0][1]
    read_confid = int(round(result[0][2], 2) * 100)
    print("===== Crop Image OCR Read - Easy ======")
    print(f'Easy OCR 결과     : {read_result}')
    print(f'Easy OCR 확률     : {read_confid}%')
    #print(" 추출결과 save : carum.txt ")
    #print(" S3 Upload path : 1iotjj/carnum")
    print("=======================================")
    #f = open(f'carnum.txt','w') # run 할때 마다 덮어쓰기 -> S3 그대로 덮어쓰기/ 파일 유지 필요없기때문
    #f.write(read_result)
    #f.close()
    #S3.upload_file(f'carnum.txt', bucket,'carnum/'+ f'carnum.txt')

# yolo ModelV1 load : 타요타요 학습된 모델 경로 : 루트 dir : ./best.pt
#input_path
input_path = './input_img/'
def YOLOV1(input_path) :
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', force_reload=True)
    # 가장 최근 생성된 차량 이미지 읽기 
    img = Image.open(recently(input_path)) # PIL
    img = img.filter(ImageFilter.GaussianBlur(radius =1))
    results = model(img, size=640) # 이미지 크롭 
    df = results.pandas().xyxy[0]
    test_crops1 = results.crop(save=False) # test_crops1 dir 생성 
    for num, crop in enumerate(test_crops1) :
        if 'plate' in crop['label'] and crop['conf'].item() * 100 > 50 :
            image = crop['im']
            im = Image.fromarray(image)   
            im.save(os.path.join(test_crops1, f'plate결과.png'), 'png',dpi=(300,300))

# yolo ModelV1 load : 타요타요 학습된 모델 경로 : 루트 dir : ./best.pt     
V1_path = './test_crops1/'
def YOLOV2(input_path) :
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', force_reload=True)
    # 가장 최근 생성된 차량 이미지 읽기 
    img = Image.open(recently(input_path)) # PIL
    img = img.filter(ImageFilter.GaussianBlur(radius =1))
    results = model(img, size=640) # 이미지 크롭 
    df = results.pandas().xyxy[0]
    test_crops1 = results.crop(save=False) # test_crops1 dir 생성 
    for num, crop in enumerate(test_crops1) :
        if 'plate' in crop['label'] and crop['conf'].item() * 100 > 50 :
            image = crop['im']
            im = Image.fromarray(image)   
            im.save(os.path.join(test_crops1, f'plate결과.png'), 'png',dpi=(300,300))
# yolo ModelV2 load : 타요타요 학습된 모델 경로 : 루트 dir : ./best.pt

        

easy_ocr(recently('./crops/'))

