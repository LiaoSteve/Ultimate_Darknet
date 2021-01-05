'''
- [x] Split trainval, train, val, test sets
- [x] Created by LiaoSteve on 2020/11/29
- [x] Change 3 parameters:
      (1) random seed
      (2) train_percent 
      (3) val_percent 
'''
import os
import random

# change 3 parameters:
random.seed(15)
train_percent = 0.7 # 0 < train percent < 1
val_percent   = 0.2

test_percent = 1 - train_percent -val_percent
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
total_images.sort()
random.shuffle(total_images)
total_txt = os.listdir(txtpath)

print(f'-------------------------------')
print(f'- [x] total_xml: {len(total_xml)}')
print(f'- [x] total_images: {len(total_images)}')
print(f'- [x] total_txt (note that classes.txt could exist, check this file!): {len(total_txt)}')

if not len(total_xml) == len(total_images): 
    w1 = input("- [warning] number of xml and images are not the same? [y/n]")
    if w1 == 'y' or w1 =='Y':
        pass
    else:
        raise RuntimeError('check xml and image')
if not len(total_images) == len(total_txt): 
    w2 = input('- [warning] number of images and txt are not the same? [y/n]')
    if w2 == 'y' or w1 =='Y':
        pass
    else:
        raise RuntimeError('check txt and image')

list = range(len(total_images))
tr = int(len(total_images) * train_percent)
va = int(len(total_images) * val_percent)
train = random.sample(list, tr) # 3825*0.7 

ftrainval = open('2007_trainval.txt', 'w')
ftest = open('2007_test.txt', 'w')
ftrain = open('2007_train.txt', 'w')
fval = open('2007_val.txt', 'w')

train_num = 0
val_num = 0
test_num = 0
val_and_test_name= []

for i in list:
    name = root + '/' + jpegfilepath + total_images[i] + '\n'
    if i in train:
        ftrain.write(name)       
        ftrainval.write(name) 
        train_num += 1                
    else:        
        val_and_test_name.append(name)

list = range(len(val_and_test_name))
val= random.sample(list,va) 

for i in list:
    name = val_and_test_name[i]
    if i in val:
        fval.write(name)
        ftrainval.write(name)
        val_num += 1
    else:        
        ftest.write(name)
        test_num += 1

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

if not len(total_images) == (train_num+test_num+val_num):
    raise RuntimeError('ERROR')

print(f'---------- DONE ----------')
print(f'- [INFO] train:val:test: = {train_percent}:{val_percent}:{test_percent:.3f}')
print(f'- [x] trainval: {train_num + val_num}')
print(f'- [x] train: {train_num}')
print(f'- [x] val: {val_num}')
print(f'- [x] test: {test_num}')



