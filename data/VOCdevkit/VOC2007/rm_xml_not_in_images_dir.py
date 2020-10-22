# Create by LiaoSteve on 2020/10/18
import os

num = 0
name_list = list()

image_names = os.listdir('./JPEGImages')
xmls_list = os.listdir('./Annotations')

image_names.sort()
xmls_list.sort()

for image_name in image_names:        
    if image_name.endswith('.jpg') or image_name.endswith('.png') or image_name.endswith('.PNG') or image_name.endswith('.JPG'):
        name_list.append(image_name[:-4])
    elif image_name.endswith('.jpeg') or image_name.endswith('.JPEG'):            
        name_list.append(image_name[:-5])
    else:            
        raise RuntimeError(f'- [x] Not the correct image format: {image_name}, delete this image and xml please.')    
del image_names

for xml in xmls_list:
    if xml.split('.xml')[0] not in name_list:        
        os.remove('./Annotations/' + xml)
        num += 1

print(f'- [x] Remove: {num} xml file')

image_names = os.listdir('./JPEGImages')
xmls_list = os.listdir('./Annotations')
print(f'- [x] JPEGImages dir: {len(image_names)} images')
print(f'- [x] Annotations dir: {len(xmls_list)} xml file')