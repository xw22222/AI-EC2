## 프로젝트 최종 CODE  해당코드 NOhub(백그라운드)돌리기 : CPU많이 잡아 먹어서 시연때만 켜기
## 머리를 씁시다 :  cron/incron/ 모니터링(와치독) 다 필요없었음.

import boto3
import os, sys
import torch
from PIL import Image, ImageFilter
import easyocr
from cloudpathlib import CloudPath
S3 = boto3.client('s3')
bucket = '1iotjj'
input_path = './input_img/'

#---------------------------------------funcions-----------------------------------------#
# 가장 최근 생성된(생성시간이 큰) 파일을 리턴하는 함수 
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

# OCR 실행 + 차량번호 저장해서 S3반환 함수
def easy_ocr (path) :
    input = recently(input_path) # 가장 최근 수신된 이미지를 받는 변수 input
    resultname = os.path.basename(input)    # 파일에서 이름만 resultname으로 가져옴/ 버킷에 올릴때만 필요 
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
    f.write(read_result) #OCR 결과부분만 쓰기 모드로 작성
    f.close()
    S3.upload_file(f'carnum.txt', bucket,'carnum/'+ f'{resultname}.txt') #S3/carnum dir에 최근 input_img.txt로 업로드

# YOLO 실행 함수 
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
#---------------------------------------funcions-----------------------------------------#
# 실행부 
# 실행 코드 무한루프
# run할때 : 루트 dir / 버킷dir 샘플 이미지 한개씩 넣어둬야됨 
while True :
    compare1 = recently(input_path)   # 먼저 현재 input_img경로의 사진중 최근 이미지를 변수에 저장
    cp = CloudPath("s3://1iotjj/media/")
    cp.download_to(input_path)        # S3버킷에서 CP : 업로드된게 없으면 루트에도 없음(덮어쓰기)     
    compare2 = recently(input_path)   # CP 후 Input_img 경로의 사진중 최근 이미지를 변수에 저장
    if compare1 == compare2 :         # 두 값이 같으면 (실행 X)
        None
    else :
        YOLO(input_path)              # CP후 값이 추가되면 다르니께 AI 모델 실행 
        easy_ocr(recently('./crops/'))


##최종 test끝나면 cp.dowload s3 경로 / easy_ocr output 경로 수정