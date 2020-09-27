import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test'), ('2007', 'trainval')]

classes = ["garbage","trash","bottle"]



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
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        
        bb = convert((w,h), b)
        if bb[0]>1 or bb[1]>1 or bb[2]>1 or bb[3]>1 :            
            raise RuntimeError(f'{image_id} bbox out of range >1')

        if bb[0]<0 or bb[1]<0 or bb[2]<0 or bb[3]<0 :            
            raise RuntimeError(f'{image_id} bbox out of range <0')
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):        
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_names = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    image_ids = list()
    for image_name in image_names:        
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_ids.append(image_name[:-4])
        elif image_name.endswith('.jpeg'):            
            image_ids.append(image_name[:-5])
        else:
            print(f'- [x] Not correct file format: {image_name}')
            raise ('error')    

    list_file = open('%s_%s.txt'%(year, image_set), 'w')

    for i in range(len(image_ids)):            
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s\n'%(wd, year, image_names[i]))
        convert_annotation(year, image_ids[i])
    list_file.close()
    print(f'- [x] {year}_{image_set} : {len(image_names)}')

print(f'- [Done] we change the xml_file(pascal VOC) to txt_file(YOLO) labels and create dataset list.')
