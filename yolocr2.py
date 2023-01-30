import os
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from PIL import Image, ImageFilter
import easyocr
import time

# 가장 최근 파일을 리턴하는 함수 : 짠거
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


def easy_ocr (path) :
    reader = easyocr.Reader(['ko', 'en'], gpu=True)
    result = reader.readtext(path)
    read_result = result[0][1]
    read_confid = int(round(result[0][2], 2) * 100)
    print("===== Crop Image OCR Read - Easy ======")
    print(f'Easy OCR 결과     : {read_result}')
    print(f'Easy OCR 확률     : {read_confid}%')
    print("=======================================")



#folder_path = './input_img/',
path = './crops'
# OCR 결과 읽는 부분 .txt파일로 저장 예정

"""
# 특정 folder 내에 있는 "가장 최근에 생성된" 파일을 리턴 
# each_file_path_and_gen_time: 각 file의 경로와, 생성 시간을 저장함
each_file_path_and_gen_time = []
for each_file_name in os.listdir(folder_path):
    # getctime: 입력받은 경로에 대한 생성 시간을 리턴
    each_file_path = folder_path + each_file_name
    each_file_gen_time = os.path.getctime(each_file_path)
    each_file_path_and_gen_time.append(
        (each_file_path, each_file_gen_time)
    )
# 가장 생성시각이 큰(가장 최근인) 파일을 리턴 
most_recent_file = max(each_file_path_and_gen_time, key=lambda x: x[1])[0]
"""
# yolo Model load : 학습모델 경로 ./best.pt
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', force_reload=True)
# 가장 최근 저장된 차량 이미지 읽기 
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
im.save(os.path.join(path, f'plate_{num}.png'), 'png',dpi=(300,300))


"""
file_list = os.listdir(path)

for num, file in enumerate(file_list):
"""
#실행부 
easy_ocr(recently('./crops/'))
