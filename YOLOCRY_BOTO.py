# NO hub 백그라운드 상시 실행 
import boto3
from cloudpathlib import CloudPath
import os, sys, time, shutil

S3 = boto3.client('s3')
bucket = '1iotjj'
crop_path = './crops'
input_path = './input_img/'

cp = CloudPath("s3://1iotjj/test_media/")

while True :
    cp.download_to(input_path)
