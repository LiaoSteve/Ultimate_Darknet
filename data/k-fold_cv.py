'''
======================================================
  This script is created by LiaoSteve on 2020/12/15  
  @ Discription: create K-fold cross validation sets 
======================================================
'''
import random
import os

# k-fold cross validation
k = 5
# random seed
random.seed(15)

path = os.getcwd() + '/'
jpegfilepath = 'VOCdevkit/VOC2007/JPEGImages/'
if not os.path.exists(jpegfilepath):
    raise RuntimeError("- [x] image path is not exist.")

total_images = os.listdir(jpegfilepath)
total_images.sort()
random.shuffle(total_images)
subset = [0 for _ in range(k)]
CV_list = []

for i in range(k):
    num = len(total_images)
    sub_num = int(num/k)
    CV_list.append([])
    if i == k-1:
        subset[i] = num - i*sub_num
    else:
        subset[i] = sub_num

for i in range(k):
    if not (i == 0):
        temp = subset[i]
        subset[i] = temp + subset[i-1] 

c = 0
for i in range(k):        
    while 1:
        CV_list[i].append(total_images[c])
        c += 1
        if c == subset[i]: 
            break
        
os.makedirs('cv_dataset', exist_ok=True)
c = 0
for j in range(k):
    dir_name = f'cv_{k}-{j+1}'
    os.makedirs(f'cv_dataset/{dir_name}', exist_ok=True)
    train = open(f"cv_dataset/{dir_name}/2007_train_cv_{k}-{j+1}.txt", "w")
    test  = open(f"cv_dataset/{dir_name}/2007_test_cv_{k}-{j+1}.txt", "w")
    for i in range(k):  
        if i == c:                           
            for image in CV_list[i]:            
                test.write(path + jpegfilepath + image + '\n')    
            continue
        for image in CV_list[i]:            
            train.write(path + jpegfilepath + image + '\n')  
    c += 1   
    test.close()
    train.close()
