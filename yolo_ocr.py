import os
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from PIL import Image, ImageFilter
import easyocr
import time

"""
- 특정 folder 내에 있는 "가장 최근에 생성된" 파일을 리턴하는 방법 
"""
folder_path = './input_img/'

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


# Model load
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n_best.pt', force_reload=True)

img = Image.open(most_recent_file) # PIL

results = model(img, size=640)
crops = results.crop(save=False)

reader = easyocr.Reader(['ko', 'en'], gpu=False)
result = reader.readtext(crops)
read_result = result[0][1]
print("===== Crop Image OCR Read - Easy ======")
print(f'Easy OCR 결과     : {read_result}')
print("=======================================")

