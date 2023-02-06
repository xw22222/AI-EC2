## 프로젝트 run 용 YOLOCR /// cron 자동실행 진행중
import boto3
import os, sys
import torch
from PIL import Image, ImageFilter
import easyocr
from cloudpathlib import CloudPath

S3 = boto3.client('s3')
bucket = '1iotjj'
input_path = './input_img/'

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
def easy_ocr (path) :
    input = recently(input_path) # 가장 최근 수신된 이미지를 받는 변수 input
    resultname = os.path.basename(input)   # 여기서 이름만 가져옴 resultname으로 # 버킷에 올릴때만 필요 

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
    f = open(f'carnum.txt','w')    # carnum으로 결과 저장 # run 할때 마다 덮어쓰기 루트파일에서
    f.write(read_result)
    f.close()
    S3.upload_file(f'carnum.txt', bucket,'test_carnum/'+ f'{resultname}.txt') #S3/carnum dir에 최근입차번호.txt로 업로드

def YOLO(path) :
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', force_reload=True)
    # 가장 최근 생성된 차량 이미지 읽기 
    img = Image.open(recently(path)) # PIL
    img = img.filter(ImageFilter.GaussianBlur(radius =1))
    results = model(img, size=640) # 이미지 크롭 
    df = results.pandas().xyxy[0]
    crops = results.crop(save=False) # ./test_crops1 dir 생성 요
    for num, crop in enumerate(crops) :
        if 'plate' in crop['label'] and crop['conf'].item() * 100 > 0:
            # if 'plate' in crop['label'] and crop['conf'].item() 
            image = crop['im']
            im = Image.fromarray(image)   
            im.save(os.path.join('./crops' , f'plate_result.png'), 'png',dpi=(300,300))
            # V1결과.png : 차량이미지에서 번호판 부분만 추출된 이미지


while True :
    compare1 = recently(input_path)
    cp = CloudPath("s3://1iotjj/test_media/")
    cp.download_to(input_path)             
    compare2 = recently(input_path)
    if compare1 == compare2 :
        print("no")
    else :
        YOLO(input_path)
        easy_ocr(recently('./crops/'))