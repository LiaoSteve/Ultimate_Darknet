# Date: 2020/10/15
# Author: LiaoSteve

import cv2
import os

cap_input = '../trash_video/trash9.MP4'
save = True
out_resize = 1
out_width = 640
out_height = 480
out_fps = 230
out_name = 'GG.avi' 


if not os.path.exists(cap_input):
    raise RuntimeError(f'- [x] cap input: {cap_input} error happened.')

cap = cv2.VideoCapture(cap_input)
cap_width = int(cap.get(3))
cap_height = int(cap.get(4))
cap_fps = int(cap.get(5))

if out_resize:
    out = cv2.VideoWriter(out_name,
            cv2.VideoWriter_fourcc(*'XVID'),
            out_fps,
            (out_width,out_height))
else:
    out = cv2.VideoWriter(out_name,
            cv2.VideoWriter_fourcc(*'XVID'),
            out_fps,
            (cap_width,cap_height))

print(f'[ Input ] size: ({cap_width},{cap_height}), fps: {cap_fps}')
if out_resize:
    print(f'[ Output ] size: ({out_width},{out_height}), fps: {out_fps} ')
else:
    print(f'[ Output ] size: ({cap_width},{cap_height}), fps: {out_fps} ')
    
print(f' video speed: x{out_fps/cap_fps}')
if not cap.isOpened():
    raise RuntimeError(f'- [x] cap input: {cap_input} error happened.')
in_count = 0

while 1:
    ret, frame = cap.read()
    if not ret:
        break    
    if out_resize:
        frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_LINEAR)        
    in_count += 1
    if save:        
        out.write(frame)            
        
print(f'input_images: {in_count}')
print(f'output_images: {in_count * cap_fps/out_fps}')
out.release()
print('Done')
