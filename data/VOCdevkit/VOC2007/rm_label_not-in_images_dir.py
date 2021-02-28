# Create by LiaoSteve on 2020/11/01
import os

num = 0
name_list = list()

image_names = os.listdir('./JPEGImages')
labels_list = os.listdir('./labels')

image_names.sort()
labels_list.sort()

for image_name in image_names:        
    if image_name.endswith('.jpg') or image_name.endswith('.png') or image_name.endswith('.PNG') or image_name.endswith('.JPG'):
        name_list.append(image_name[:-4])
    elif image_name.endswith('.jpeg') or image_name.endswith('.JPEG'):            
        name_list.append(image_name[:-5])
    else:            
        raise RuntimeError(f'- [x] Not the correct image format: {image_name}, delete this image and xml please.')    
del image_names

for label in labels_list:    
    if label.split('.txt')[0] not in name_list:                    
        os.remove('./labels/' + label)
        print(f'remove: {label}')  
        num += 1

print(f'- [x] Remove: {num} label file')

image_names = os.listdir('./JPEGImages')
labels_list = os.listdir('./labels')
print(f'- [x] JPEGImages dir: {len(image_names)} images')
print(f'- [x] labels dir: {len(labels_list)} txt file')
