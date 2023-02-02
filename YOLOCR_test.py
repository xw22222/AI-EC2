## YOLO 모델 V2 Run용 
import os, sys, time
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from PIL import Image, ImageFilter
import easyocr
import pytesseract
import boto3

#S3 = boto3.client('s3')
#bucket = '1iotjj'
#path = './crops'

# 가장 최근 생성된 파일을 리턴하는 함수
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


# OCR 결과 읽고 차량번호 저장해서 S3반환 함수   
def easy_ocr_origin (path) :
    reader = easyocr.Reader(['ko'], gpu=True)
    result = reader.readtext(path)
    read_result = result[0][1]
    read_confid = int(round(result[0][2], 2) * 100)
    print("===== Crop Image OCR Read - Easy ======")
    print(f'이전 모델 YOLO V1 Easy OCR 결과     : {read_result}')
    print(f'이전 모델 YOLO V1 Easy OCR 확률     : {read_confid}%')
    #print(" 추출결과 save : carum.txt ")
    #print(" S3 Upload path : 1iotjj/carnum")
    print("=======================================")
    #f = open(f'carnum.txt','w') # run 할때 마다 덮어쓰기 -> S3 그대로 덮어쓰기/ 파일 유지 필요없기때문
    #f.write(read_result)
    #f.close()
    #S3.upload_file(f'carnum.txt', bucket,'carnum/'+ f'carnum.txt')  boto3 버킷 업로드

def easy_ocr_new (path) :
    reader = easyocr.Reader(['ko'], gpu=True)
    result = reader.readtext(path)
    read_result = result[0][1]
    read_confid = int(round(result[0][2], 2) * 100)
    print("===== Crop Image OCR Read - Easy ======")
    print(f'YOLO V1 + V2 + Easy OCR 결과     : {read_result}')
    print(f'YOLO V1 + V2 + Easy OCR 확률     : {read_confid}%')
    #print(" 추출결과 save : carum.txt ")
    #print(" S3 Upload path : 1iotjj/carnum")
    print("=======================================")
    #f = open(f'carnum.txt','w') # run 할때 마다 덮어쓰기 -> S3 그대로 덮어쓰기/ 파일 유지 필요없기때문
    #f.write(read_result)
    #f.close()
    #S3.upload_file(f'carnum.txt', bucket,'carnum/'+ f'carnum.txt')  boto3 버킷 업로드


#최초 이미지 path : input_img
V1_path = './input_img/'
V2_input_path = './test_crops1/'
V2_result_path = './test_crops2/'

# yolo ModelV1 load : 타요타요 학습된 모델 경로 : 루트 dir : ./best.pt
def YOLOV1(path) :
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./V2.pt', force_reload=True)
    # 가장 최근 생성된 차량 이미지 읽기 
    img = Image.open(recently(path)) # PIL
    img = img.filter(ImageFilter.GaussianBlur(radius =1))
    results = model(img, size=640) # 이미지 크롭 
    df = results.pandas().xyxy[0]
    crops = results.crop(save=False) # ./test_crops1 dir 생성 요
    for num, crop in enumerate(crops) :
        if 'plate' in crop['label'] and crop['conf'].item() * 100 > 50 :
            image = crop['im']
            im = Image.fromarray(image)   
            im.save(os.path.join(V2_input_path , f'V2결과.png'), 'png',dpi=(300,300))
            # V1결과.png : 차량이미지에서 번호판 부분만 추출된 이미지

# 1차 crop된 이미지 path : test_crops1

# yolo ModelV2 load : 2차 모델 루트 dir : ./yolov5s.pt // 준호님이 주신 프로젝트에서 : runs/train/exp2/weights/best.pt뽑아서
# 배포 프로젝트(여기)./ 루트경로에 삽입 -> name : V2.pt
def YOLOV2(path) :
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./V2.pt', force_reload=True)
    # 가장 최근 생성된 차량 이미지 읽기 
    img = Image.open(recently(path)) # PIL
    img = img.filter(ImageFilter.GaussianBlur(radius =1))
    results = model(img, size=640) # 이미지 크롭 
    df = results.pandas().xyxy[0]
    crops = results.crop(save=False) # ./test_crops2 dir 생성 요
    for num, crop in enumerate(crops) :
        if 'plate' in crop['label'] and crop['conf'].item() * 100 > 50 :
            image = crop['im']
            im = Image.fromarray(image)   
            im.save(os.path.join(V2_result_path , f'V2결과.png'), 'png',dpi=(300,300))
            # V2결과.png : 차량이미지에서 번호판 부분만 추출된 이미지에서 숫자를 검출 하는거 까지


# 실행부

YOLOV1(V1_path)
easy_ocr_origin(recently(V2_input_path))
YOLOV2(V2_result_path)
easy_ocr_new(recently(V2_result_path))

