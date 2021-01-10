'''
@ Author: Yu-Hsien Liao (LiaoSteve)
@ Description: copy the validation set to \n
        'mAP/input/images-optional' dir to calculate the mAP
'''
import cv2
import os
from pathlib import Path
from tqdm import tqdm 

data_path = Path('data/obj.data')
output_image_dir = Path('mAP/input/images-optional')
os.makedirs(output_image_dir, exist_ok=True)

# get the images path from .data file
with open(data_path,'r') as f:
    data = f.readlines()

for line in data:
    if 'valid' in line:            
        data = line.split(' ')[-1]            
        data = data.split('\n')[0]

# get all image path
with open(data,'r') as f:
    image_list = f.readlines()

for i in tqdm(range(len(image_list))): 
    image = image_list[i]
    image = image.split('\n')[0]  
    frame = cv2.imread(str(image))
    path, name = os.path.split(image)  
    name, _ = name.split('.')  
    name = name + '.jpg'
    cv2.imwrite(str(output_image_dir/Path(name)), frame)
print('Done')
