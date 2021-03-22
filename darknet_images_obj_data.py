"""
Author: Yu-Hsien Liao(LiaoSteve)
Date: 2021/3/3
Description: detect the objects in the image sets
"""
from ctypes import *
import random
import os
import cv2
import darknet
import argparse

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")   

    parser.add_argument("--save_dir", type=str, default="./predict_image/2_best/",
                        help="path to save detection images")

    parser.add_argument("--weights", default="./backup/yolov4_8_best.weights",
                        help="yolo weights path") 

    parser.add_argument("--config_file", default="./cfg/yolov4_8.cfg",
                        help="path to config file")

    parser.add_argument("--data_file", default="./data/obj.data",
                        help="path to data file")

    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")    

    parser.add_argument("--iou_thresh", type=float, default=.5,
                        help="nms: remove detections with iou higher this value") 
    return parser.parse_args()


def check_arguments_errors(args):    
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    assert 0 < args.iou_thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    os.makedirs(args.save_dir, exist_ok=1)


if __name__ == '__main__':
    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )

    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    darknet_image = darknet.make_image(darknet_width, darknet_height, 3)    
    info = dict()   
    sets = []

    # get the valid path
    with open(args.data_file,'r') as f:
        data = f.readlines()

    for line in data:
        if 'train' in line:            
            line = line.split(' ')[-1]            
            line = line.split('\n')[0]
            sets.append(('train',line))
            break

    for line in data:
        if 'valid' in line:            
            line = line.split(' ')[-1]            
            line = line.split('\n')[0]
            sets.append(('valid',line))
            break

    for types, data in sets:
        # get all image path
        with open(data,'r') as f:
            image_list = f.readlines()

        save_dir = args.save_dir + types +'/'
        os.makedirs(save_dir, exist_ok=1)
        temps = list()      
        images = list()        

        for filename in image_list:
            filename = filename.strip('\n')
            temps.append(filename)

        for filename in temps:
            if filename.endswith('jpg') or filename.endswith('png')\
                or filename.endswith('jpeg') or filename.endswith('JPG')\
                or filename.endswith('JPEG') or filename.endswith('PNG'):
                images.append(filename)
            else:        
                raise RuntimeError(f'notice that {filename} image format are not accepted(.jpg, .png, .jpeg)')
        
        for image in images:        
            frame = cv2.imread(image)          
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                    interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh, nms=args.iou_thresh)
            frame = darknet.draw_boxes(detections, frame, class_colors, darknet_width)
            cv2.imwrite(save_dir + image.split('/')[-1], frame)
            print(f'- [x] save image {image} to {save_dir}')
        info[types]= len(images)       
        del temps
        del images
    print(info)
