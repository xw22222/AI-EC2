## 프로젝트 run 용 YOLOCR /// cron 자동실행 진행중
import boto3
from cloudpathlib import CloudPath

S3 = boto3.client('s3')
bucket = '1iotjj'
crop_path = './crops'
input_path = './input_img/'
cp = CloudPath("s3://1iotjj/test_media/")
cp.download_to(input_path)
