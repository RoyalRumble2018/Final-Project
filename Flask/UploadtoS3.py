#Upload Models.zip to S3 !!!!


import logging 
import os
import zipfile
import boto.s3
import sys
from boto.s3.key import Key
import time
import datetime


accessKey='AKIAIW32EIXID5G7UCYA'
secretAccessKey='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'
inputLocation='USWest2'

if not accessKey or not secretAccessKey:
    logging.warning('Access Key and Secret Access Key not provided!!')
    print('Access Key and Secret Access Key not provided!!')
    exit()

AWS_ACCESS_KEY_ID = accessKey
AWS_SECRET_ACCESS_KEY = secretAccessKey

try:
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY)

    print("Connected to S3")

except:
    logging.info("Amazon keys are invalid!!")
    print("Amazon keys are invalid!!")
    exit()

try:   
    bucket = conn.get_bucket('modelsads')
    print(bucket)
except:
    logging.info("Amazon keys are invalid!!")
    print("Amazon keys are invalid!")
    exit()


def percent_cb(complete, total):
    sys.stdout.write('.')
    sys.stdout.flush()

zipfile = 'Models.zip'
k = Key(bucket)
k.key = 'Models'
k.set_contents_from_filename(zipfile,cb=percent_cb, num_cb=10)
print("Zip File successfully uploaded to S3") 







