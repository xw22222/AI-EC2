## 프로젝트 run 용 YOLOCR : 이거만 짜면됨 
import boto3
import os, sys, time
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from PIL import Image, ImageFilter
import easyocr
import pytesseract

S3 = boto3.client('s3')
bucket = '1iotjj/kjtest'
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
    print("===========번호.txt 저장 : /txresult=============")
    f = open(f'{read_result}.txt','w')
    f.write(read_result)
    f.close()
        # 여기서 boto3 바로 쓰면 굳이 저장안해도됨 서버에 ?


#크롭 이미지 저장된거 덮어쓰기 됨?


# yolo Model load : 타요타요 학습된 모델 경로 : 루트 dir : ./best.pt
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', force_reload=True)

# 가장 최근 생성된 차량 이미지 읽기 
img = Image.open(recently('./input_img/')) # PIL
img = img.filter(ImageFilter.GaussianBlur(radius =1))

# 이미지 크롭 
results = model(img, size=640)
df = results.pandas().xyxy[0]
crops = results.crop(save=False)
# conf = (crop[0]['conf'].item() * 100)

for num, crop in enumerate(crops) :
    if 'plate' in crop['label'] and crop['conf'].item() * 100 > 50 :
        image = crop['im']
        im = Image.fromarray(image)   
        im.save(os.path.join(path, f'plate_{img}.png'), 'png',dpi=(300,300))

#가장 최근 생성된 Crops 결과 이미지 easy_ocr 함수 읽기 
#실행부 
easy_ocr(recently('./crops/'))
#s3.upload_file(recently('./txtresult'), bucket,recently('./txtresult'))
