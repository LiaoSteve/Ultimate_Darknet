'''
=======================================================================
  This script is created by LiaoSteve on 2021/1/5  
  @ Discription: create Nested Cross-Validation sets or k-fold CV sets
  @ python version >= 3.6 (f-string)  
=======================================================================
'''
import random
import os

# outer fold k > 0
k = 5
if not k > 1: raise RuntimeError('change k > 1') 

# inner fold k2 >= 0
k2 = 2
if k < 0: raise RuntimeError('change k2 >= 0') 

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

CV_list = [] # outer loop
inner_CV_list = [] # inner loop

# divide all data into k-fold
for i in range(k):
    num = len(total_images)
    sub_num = int(num/k)
    CV_list.append([])
    if i == k-1:
        subset[i] = num - i*sub_num
    else:
        subset[i] = sub_num

# record the ith image of each fold
for i in range(k):
    if not (i == 0):
        temp = subset[i]
        subset[i] = temp + subset[i-1] 

# append the images to each fold
c = 0
for i in range(k):        
    while 1:
        CV_list[i].append(total_images[c])
        c += 1
        if c == subset[i]: 
            break
# ====== create nested cv ========
os.makedirs('nested_cv_dataset', exist_ok=True)
c = 0
c3 = 0
for j in range(k):
    trainval_list = []
    dir_name = f'outer_cv_{k}-{j+1}'
    os.makedirs(f'nested_cv_dataset/{dir_name}', exist_ok=True)
    trainval = open(f"nested_cv_dataset/{dir_name}/train_cv_{k}-{j+1}.txt", "w")
    test  = open(f"nested_cv_dataset/{dir_name}/test_cv_{k}-{j+1}.txt", "w")
    for i in range(k):  
        if i == c:                           
            for image in CV_list[i]:            
                test.write(path + jpegfilepath + image + '\n')    
            continue
        for image in CV_list[i]:            
            trainval.write(path + jpegfilepath + image + '\n') 
            trainval_list.append(image)  

    # divide inner data into k-fold
    subset = [0 for _ in range(k2)]
    for i in range(k2):
        num = len(trainval_list)
        sub_num = int(num/k2)
        inner_CV_list.append([])
        if i == k2-1:
            subset[i] = num - i*sub_num
        else:
            subset[i] = sub_num

    # record the ith image of each fold
    for i in range(k2):
        if not (i == 0):
            temp = subset[i]
            subset[i] = temp + subset[i-1] 

    # append the images to each fold
    c2 = 0
    for i in range(k2):        
        while 1:
            inner_CV_list[i].append(trainval_list[c2])
            c2 += 1
            if c2 == subset[i]: 
                break

    for n in range(k2):
        inner_dir_name = f'inner_cv_{k2}-{n+1}'
        os.makedirs(f'nested_cv_dataset/{dir_name}/{inner_dir_name}', exist_ok=True) 
        inner_train = open(f"nested_cv_dataset/{dir_name}/{inner_dir_name}/train.txt","w")
        inner_val = open(f"nested_cv_dataset/{dir_name}/{inner_dir_name}/val.txt","w")
        for m in range(k2):            
            if m == c3:                           
                for image in inner_CV_list[m]:            
                    inner_val.write(path + jpegfilepath + image + '\n')    
                continue
            for image in inner_CV_list[m]:            
                inner_train.write(path + jpegfilepath + image + '\n') 
        c3 += 1
        inner_train.close()
        inner_val.close()
    c += 1   
    test.close()
    trainval.close()