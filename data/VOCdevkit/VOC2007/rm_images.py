# Create by LiaoSteve on 2020/10/18
import os

num = 0
path = os.path.abspath('.')
images_list = os.listdir('./JPEGImages')
xmls_list = os.listdir('./Annotations')
images_list.sort()
xmls_list.sort()
x = input('- [x] Delete a half of images in JPEGImages dir ? [y/n] ')

if not x =='y':
    raise RuntimeError(f'- [x] Be careful ~')

for i in range(len(images_list)):
    if i % 2 == 0:
        image_name = images_list[i]  
        xml_name = xmls_list[i]      
        os.remove('./JPEGImages/' + image_name)
        os.remove('./Annotations/' + xml_name)
        num += 1

print(f'remove: {num} images')
print(f'remove: {num} xml file')