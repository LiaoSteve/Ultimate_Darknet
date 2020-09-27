import os
import random
 
trainval_percent = 0.1
train_percent = 0.9

xmlfilepath = 'Annotations'
txtsavepath = 'ImageSets/Main'
jpegfilepath = 'JPEGImages'

random.seed(10)

total_xml = os.listdir(xmlfilepath)
total_images = os.listdir(jpegfilepath)

num = len(total_xml)
image_num = len(total_images)

print(f'- [x] total_xml: {num}')
print(f'- [x] total_images: {image_num}')

if num != image_num:    
    raise RuntimeError("- [x] number of xml and images are not the same.")
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
 
if not os.path.exists(txtsavepath):
    print('not exist...{}'.format(txtsavepath))
    os.makedirs(txtsavepath)
 
ftrainval = open('ImageSets/Main/trainval.txt', 'w')
ftest = open('ImageSets/Main/test.txt', 'w')
ftrain = open('ImageSets/Main/train.txt', 'w')
fval = open('ImageSets/Main/val.txt', 'w')
 
for i in list:
    name = total_images[i] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)
 
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

print(f'- [Done] Create dataset(txtfile) to {txtsavepath}')
print(f'- [x] Please run voc_label.py continuously ...')
