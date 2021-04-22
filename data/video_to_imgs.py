# Author: LiaoSteve
# Date: 2020/9/5
# This script can turn video frames to images.

import os 
import cv2

input_path = 'FPV.avi'
out_dir    = 'images_FPV'

os.makedirs(out_dir, exist_ok=True)
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(5))
all_frame = int(cap.get(7))
count = 0
num = 0
desired_fps = 2 # get image per second
out_fps = int(fps/desired_fps)

while True:
    ret, img = cap.read()
    count += 1    
    if ret:
        if count % out_fps == 0:
            num += 1
            cv2.imwrite(out_dir+'/'+out_dir+'_'+str(num)+'.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            #print(f"{count}/{all_frame}")
    else: break
print(f'- [x] Done: save {num} images')

