## 프로젝트 run 용 YOLOCR /// cron 자동실행 진행중
import boto3, easyocr, torch
import os, sys, time, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=UserWarning)
from PIL import Image, ImageFilter
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

S3 = boto3.client('s3')
bucket = '1iotjj'
crop_path = './crops'
input_path = './input_img/'

if __name__ == "__main__":
    patterns = ["*"]
    ignore_patterns = None
    ignore_directories = False
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)

def on_created(event):
    event.YOLOV(input_path) 
    event.easy_ocr(recently(crop_path))

    """
    if event.src_path == True :
        YOLOV(input_path) 
        easy_ocr(recently(crop_path))
        #print(f"hey, {event.src_path} has been created!")
    else :
        print("입차 차량 없음")
    """    
def recently(folder_path) : # 가장 최근 생성된 파일을 리턴하는 함수 
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
    reader = easyocr.Reader(['ko'], gpu=True)
    result = reader.readtext(path)
    read_result = result[0][1]
    read_confid = int(round(result[0][2], 2) * 100)
    print("===== Crop Image OCR Read - Easy ======")
    print(f'Easy OCR 결과     : {read_result}')
    print(f'Easy OCR 확률     : {read_confid}%')
    print("Easy ocr 결과 save : carum.txt ")
    print("AWS S3 Upload path : 1iotjj/carnum")
    print("=======================================")
    f = open(f'{read_result}.txt','w')
    f = open(f'carnum.txt','w') # run 할때 마다 덮어쓰기 -> S3 그대로 덮어쓰기/ 파일 유지 필요 없음
    f.write(read_result)
    f.close()
    S3.upload_file(f'carnum.txt', bucket,'carnum/'+ f'carnum.txt') #S3/carnum dir에 carnum.txt로 업로드 

def YOLOV(path) :
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
            im.save(os.path.join(crop_path , f'크롭결과.png'), 'png',dpi=(300,300))
            # V1결과.png : 차량이미지에서 번호판 부분만 추출된 이미지

my_event_handler.on_created = on_created

path = './input_img/'

go_recursively = True

my_observer = Observer()

my_observer.schedule(my_event_handler, path, recursive=go_recursively)

my_observer.start()
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    my_observer.stop()
    my_observer.join()