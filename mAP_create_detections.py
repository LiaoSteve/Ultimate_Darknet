# --------------------------------------------------------
# Author: Yu-Hsien Liao (LiaoSteve)
# Date: 2021/01/09
# python version >=3.6 
# Description: 1. save the detection results to txt file.  
#              2. save marked detection images to save_image_dir
# For example:
#     There are 4 detections in 001.jpg
#     In 001.txt:
#     <classname> score left top right bottom
#     mouth 0.8330048 135 400 221 435
#     nose 0.95975435 132 339 227 382
#     eye 0.50498945 208 269 286 302
#     eye 0.5545531 78 269 154 302
# --------------------------------------------------------

from ctypes import *
import random
import os
import cv2
import darknet
import argparse
from pathlib import Path
from tqdm import tqdm 

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")       

    parser.add_argument(
            "--weights", 
            default="backup/cv5-1/yolov4_8/yolov4_8_7000.weights",
            help="yolo weights path") 

    parser.add_argument(
            "--config_file", 
            default="cfg/yolov4_8.cfg",
            help="path to config file")

    parser.add_argument(
            "--data_file", 
            default="data/obj.data",
            help="path to data file")

    parser.add_argument(
            "--save_image_dir", 
            type=str, 
            default="",
            help="path to save detection images")

    parser.add_argument(
            "--detection_dir", 
            type=str, 
            default="mAP/input/detection-results/",
            help="path to save detection txt file")

    parser.add_argument(
            "--thresh", 
            type=float, 
            default=.25,
            help="remove detections with confidence below this value")    

    parser.add_argument(
            "--iou_thresh", 
            type=float, 
            default=.45,
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
    if args.save_image_dir:
        os.makedirs(args.save_image_dir, exist_ok=1)
    if args.detection_dir:
        os.makedirs(args.detection_dir, exist_ok=1)

if __name__ == '__main__':
    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    # class_colors['garbage'] = (255,0,0)
    # class_colors['bottle'] = (0,255,0)

    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    darknet_image = darknet.make_image(darknet_width, darknet_height, 3)

    # get the valid path
    with open(args.data_file,'r') as f:
        data = f.readlines()
    
    for line in data:
        if 'valid' in line:            
            data = line.split(' ')[-1]            
            data = data.split('\n')[0]

    # get all image path
    with open(data,'r') as f:
        image_list = f.readlines()
    for i in tqdm(range(len(image_list))):
        image = image_list[i]
        image = image.split('\n')[0]
        frame = cv2.imread(image)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(
            network, 
            class_names, 
            darknet_image, 
            thresh=args.thresh, 
            nms=args.iou_thresh)        

        path, name = os.path.split(image)

        if args.save_image_dir:
            frame = darknet.draw_boxes(detections, frame, class_colors, darknet_width)
            cv2.imwrite(args.save_image_dir + 'out_' + name, frame)
            print(f'- [x] save image {name} to {args.save_image_dir}')

        if args.detection_dir:
            # save detections to  txt file
            txt_name, _format = name.split('.')
            with open(Path(args.detection_dir + '/' + txt_name+'.txt'), 'w') as f:                
                for label, confidence, bbox in detections:
                    left, top, right, bottom = \
                            darknet.bbox2points(bbox, darknet_width, frame.shape)
                    f.write(f'{label} {round(float(confidence)/100,6)} {left} {top} {right} {bottom}\n')
    print('Done')
