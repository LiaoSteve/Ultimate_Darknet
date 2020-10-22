'''
*  Adapted by LiaoSteve on 2020/10/18
*  change 3 parameters:
        random.seed()
        trainval_percent 
        train_percent 
'''
import os
import random

# change 3 parameters:
random.seed(10)
trainval_percent = 0.2
train_percent = 0.8

root = os.getcwd()
xmlfilepath = 'VOCdevkit/VOC2007/Annotations/'
txtpath = 'VOCdevkit/VOC2007/labels/'
jpegfilepath = 'VOCdevkit/VOC2007/JPEGImages/'

if not os.path.exists(xmlfilepath):
    raise RuntimeError("- [x] xml path is not exist.")
if not os.path.exists(jpegfilepath):
    raise RuntimeError("- [x] image path is not exist.")
if not os.path.exists(txtpath):
    raise RuntimeError("- [x] txt path is not exist.")

total_xml = os.listdir(xmlfilepath)
total_images = os.listdir(jpegfilepath)
total_txt = os.listdir(txtpath)

print(f'------------------------ \n- [INFO] train:trainval = {train_percent}:{trainval_percent}')
print(f'- [x] total_xml: {len(total_xml)}')
print(f'- [x] total_images: {len(total_images)}')
print(f'- [x] total_txt (note that classes.txt could exist, check this file!): {len(total_txt)}')

if not len(total_xml) == len(total_images) and len(total_images) == len(total_txt):  
    raise RuntimeError("- [x] number of xml and images and txt are not the same.")

list = range(len(total_images))
tv = int(len(total_images) * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
 
ftrainval = open('2007_trainval.txt', 'w')
ftest = open('2007_test.txt', 'w')
ftrain = open('2007_train.txt', 'w')
fval = open('2007_val.txt', 'w')

train_num = 0
val_num = 0
test_num = 0

for i in list:
    name = root + '/' + jpegfilepath + total_images[i] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
            test_num += 1
        else:
            fval.write(name)
            val_num += 1
    else:
        ftrain.write(name)
        train_num += 1
 
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

if not len(total_images) == (train_num+test_num+val_num):
    raise RuntimeError('ERROR')

print(f'----- DONE ------')
print(f'- [x] train: {train_num}')
print(f'- [x] trainval: {test_num + val_num}')
print(f'- [x] validation: {val_num}')
print(f'- [x] test: {test_num}')


