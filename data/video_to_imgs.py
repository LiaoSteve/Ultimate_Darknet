# Author: LiaoSteve
# Date: 2020/9/5
# This script can turn video frames to images.

import os 
import cv2

input_path = '123.mp4'
out_dir    = 'images'

os.makedirs(out_dir, exist_ok=True)
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(5))

count = 0
num = 0
out_fps = 2

while True:
    ret, img = cap.read()
    count += 1    
    if ret:
        if count % fps < out_fps:
            num += 1
            cv2.imwrite(out_dir+'/out_'+str(num)+'.jpg', img)
            print(num)
    else: break
print('- [x] Done')
