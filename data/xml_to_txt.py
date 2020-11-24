'''
0. Adapted by LiaoSteve on 2020/10/19
1. You should use 'labelimg' sofware to label xml files
2. Turn .xml files in Annotations dir into .txt files in labels dir(be created automatically)
'''
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('2007', 'dont_care')]

classes = ['garbage','bottle']

num = 0
P_object = [0 for _ in classes]
N_sample = 0
difficult_object = 0
not_my_desired_object = 0

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0 
    y = (box[2] + box[3])/2.0 
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    global N_sample
    global not_my_desired_object
    global difficult_object
    global P_object
    global num 
    try:
        in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    except Exception as e:        
        N_sample += 1
        print(e)
        return

    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    if root.find('object') is None:        
        N_sample += 1        

    for obj in root.iter('object'):        
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        if cls not in classes or int(difficult) == 1:     
            print(f'Skip {cls} object, difficult {int(difficult)}')       
            if cls not in classes:                
                not_my_desired_object += 1
            if int(difficult) == 1:                
                difficult_object += 1
            continue

        cls_id = classes.index(cls)        
        P_object[cls_id] += 1
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))        
        bb = convert((w,h), b)

        if bb[0]>1 or bb[1]>1 or bb[2]>1 or bb[3]>1 :            
            raise RuntimeError(f'{image_id} bbox out of range >1')

        if bb[0]<0 or bb[1]<0 or bb[2]<0 or bb[3]<0 :            
            raise RuntimeError(f'{image_id} bbox out of range <0')

        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')    
    num += 1

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):        
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))    
   
    image_names = os.listdir('VOCdevkit/VOC%s/JPEGImages/'%(year))

    for image_name in image_names:        
        if image_name.endswith('.jpg') or image_name.endswith('.png') or image_name.endswith('.PNG') or image_name.endswith('.JPG'):
            pass
        elif image_name.endswith('.jpeg') or image_name.endswith('.JPEG'):            
            pass
        else:            
            raise RuntimeError(f'- [x] Not the correct image format: {image_name}, delete this image and xml please.')    
    del image_names

    image_ids = os.listdir('./VOCdevkit/VOC%s/JPEGImages/'%(year))  
    xml_ids = os.listdir('./VOCdevkit/VOC%s/Annotations/'%(year))
    
    image_ids.sort()
    xml_ids.sort()

    if not len(image_ids) ==len(xml_ids):
        x = input(f'number of images {len(image_ids)} and .xml files {len(xml_ids)} not equal. Continue? [y/n]')
        if x == 'y' or x == 'Y':
            pass
        else:
            raise RuntimeError('Check please!')
    for image_id in image_ids:        
        convert_annotation(year, os.path.splitext(image_id)[0])   

print('---- DONE ----')   
print(f'- [x] Turn {len(xml_ids)} .xml files to {num} .txt files')
print(f'- [x] JPEGImages: {len(image_ids)} images')
n_label = os.listdir('VOCdevkit/VOC2007/labels/')
print(f'- [x] label: {len(n_label)}')
print(f'---- Info ----')
for i in range(len(classes)):
    print(f'- [x] {classes[i]}: {P_object[i]} objects')
print(f'- [x] Negative image: {N_sample}')
print(f'- [x] difficult object: {difficult_object}')
print(f'- [x] not my desired object: {not_my_desired_object}')


