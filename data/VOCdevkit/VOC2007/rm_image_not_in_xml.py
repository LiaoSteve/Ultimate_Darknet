# Create by LiaoSteve on 2020/10/18
import os

num = 0
name_list = list()

image_names = os.listdir('./JPEGImages')
xmls_list = os.listdir('./Annotations')

image_names.sort()
xmls_list.sort()

print(f'- [x] JPEGImages dir: {len(image_names)} images')
print(f'- [x] Annotations dir: {len(xmls_list)} xml file')
for xml_name in xmls_list:    
    if xml_name.endswith('.xml'):
        name_list.append(xml_name[:-4])

k = input('delete images not in xml dir? [y/n]')
if not k == 'y':
    raise RuntimeWarning('exit')

for image_name in image_names:
    name = image_name.split('.')[0]
    if name not in name_list:        
        os.remove('./JPEGImages/' + image_name)
        num += 1

print(f'- [x] Remove: {num} xml file')

image_names = os.listdir('./JPEGImages')
xmls_list = os.listdir('./Annotations')
print(f'- [x] JPEGImages dir: {len(image_names)} images')
print(f'- [x] Annotations dir: {len(xmls_list)} xml file')
