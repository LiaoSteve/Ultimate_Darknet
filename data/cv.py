#################################
####   Created by LiaoSteve #####
#################################
import random
import os

jpegfilepath = 'VOCdevkit/VOC2007/JPEGImages/'
if not os.path.exists(jpegfilepath):
    raise RuntimeError("- [x] image path is not exist.")
random.seed(15)
total_images = os.listdir(jpegfilepath)
total_images.sort()
random.shuffle(total_images)
print('gg')