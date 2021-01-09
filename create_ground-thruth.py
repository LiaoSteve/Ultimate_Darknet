# --------------------------------------------------------
# Author: Yu-Hsien Liao (LiaoSteve)
# Date: 2021/01/09
# python version >=3.6 
# Description: save the ground thruth results to txt file.  
#  
# For example:
#     There are 4 G.T. in 001.jpg
#     In 001.txt:
#     <classname> left top right bottom
#     mouth 135 400 221 435
#     nose 132 339 227 382
#     eye 208 269 286 302
#     eye 78 269 154 302
# --------------------------------------------------------
import os
from pathlib import Path
import cv2

def yolo_2_voc(x, y, w, h, img_w, img_h):
    """
    x, y : float bbox's center\n
    w, h : float bbox's width and height\n
    img_w, img_h : integer image's width and height\n
    return left, top, right, bottom
    """
    A = 2 * img_w * x 
    B = 2 * img_h * y
    C = img_w * w
    D = img_h * h

    left   = (A-C)/2
    top    = (B-D)/2
    right  = (A+C)/2
    bottom = (B+D)/2
    # in the official VOC challenge the top-left 
    # pixel in the image has coordinates (1;1)
    return int(left)+1, int(top)+1, int(right)+1, int(bottom)+1

if __name__ == '__main__':

    data_file = Path('data/obj.data')
    data_classes = Path('data/obj.names')
    label_dir = Path('data/VOCdevkit/VOC2007/labels')
    output_dir = Path('mAP/input/ground-truth')
    os.makedirs(output_dir, exist_ok=1)
    # get the classes
    classes = []
    with open(data_classes, 'r') as f:
        data = f.readlines()
    for class_ in data:
        classes.append(class_.split('\n')[0])
    
    # get the valid path
    with open(data_file,'r') as f:
        data = f.readlines()
    
    for line in data:
        if 'valid' in line:            
            data = line.split(' ')[-1]            
            data = data.split('\n')[0]

    # get all image path
    with open(data,'r') as f:
        image_list = f.readlines()
    
    for image in image_list:
        image = image.split('\n')[0]        
        frame = cv2.imread(image)
        img_h, img_w, _ = frame.shape

        path, name = os.path.split(image)
        txt_name = name.split('.')[0] + '.txt'      
       
        with open(label_dir/txt_name,'r') as f:
            gts = f.readlines()   

        with open(output_dir/txt_name,'w') as f:
            for gt in gts:
                gt = gt.split('\n')[0]
                class_id, x, y, w, h = gt.split(' ')                
                left, top, right, bottom \
                        = yolo_2_voc(float(x),float(y),float(w),float(h),img_w,img_h)
                f.write(f'{classes[int(class_id)]} {left} {top} {right} {bottom}\n')

    print('Done')   