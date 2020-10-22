# Created by LiaoSteve on 2020-09-27
import os

head_name = 'data1_7_12'

xml_list = sorted(os.listdir('./Annotations'))
image_list = sorted(os.listdir('./JPEGImages'))

if len(xml_list) == len(image_list):
    num = len(xml_list)
else :
    raise RuntimeError('number of xml_list and image_list are not equal.')
# rename image list
os.chdir('./JPEGImages')
for i in range(num):
    temp = image_list[i]
    if temp.endswith('.jpg'):
        temp = '.jpg'
    elif temp.endswith('.png'):
        temp = '.png'
    elif temp.endswith('.jpeg'):
        temp = '.jpeg'
    elif temp.endswith('.JPG'):
        temp = '.JPG'
    elif temp.endswith('.PNG'):
        temp = '.PNG'
    elif temp.endswith('.JPEG'):
        temp = '.JPEG'
    else:
        raise RuntimeError('images format is not .jpg .jpeg or png')    
    os.rename(image_list[i], head_name + '_' + str(i)+temp)

# rename xml list
os.chdir('./../Annotations')

for i in range(num):
    os.rename(xml_list[i], head_name + '_' + str(i)+'.xml')

print('- [Done] convert images and xml names to numbers')