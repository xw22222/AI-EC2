# NO hub으로  백그라운드 상시 실행 
import boto3
from cloudpathlib import CloudPath

S3 = boto3.client('s3')
bucket = '1iotjj'
input_path = './input_img/'

cp = CloudPath("s3://1iotjj/test_media/")
cp.download_to(input_path)
