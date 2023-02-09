# ubuntu : ./input_img에 샘플 이미지 넣는용도 
import boto3
from cloudpathlib import CloudPath

S3 = boto3.client('s3')
bucket = '1iotjj'
input_path = './input_img/'

cp = CloudPath("s3://1iotjj/media/")
cp.download_to(input_path)
